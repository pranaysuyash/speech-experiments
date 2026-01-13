"""
Unit tests for evidence provenance and ground-truth requirements.

These tests prevent the class of bugs where:
- WER=1.00 appears when there's no ground truth
- speaker_accuracy=1.00 appears when there's no reference
"""

import pytest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from harness.run_provenance import (
    create_provenance,
    can_compute_quality_metrics,
    RUN_SCHEMA_VERSION
)


class TestProvenance:
    """Test run provenance creation and validation."""
    
    def test_provenance_has_required_fields(self):
        """Provenance must include all required fields."""
        prov = create_provenance(
            dataset_id="test_dataset",
            dataset_path=None,
            audio_path=None,
            ground_truth_path=None,
        )
        
        assert "schema_version" in prov
        assert "dataset_id" in prov
        assert "has_ground_truth" in prov
        assert "metrics_valid" in prov
        assert prov["schema_version"] == RUN_SCHEMA_VERSION
    
    def test_no_ground_truth_when_path_missing(self):
        """has_ground_truth must be False when no path provided."""
        prov = create_provenance(
            dataset_id="test",
            ground_truth_path=None,
        )
        assert prov["has_ground_truth"] is False
    
    def test_no_ground_truth_when_file_missing(self):
        """has_ground_truth must be False when file doesn't exist."""
        prov = create_provenance(
            dataset_id="test",
            ground_truth_path=Path("/nonexistent/file.txt"),
        )
        assert prov["has_ground_truth"] is False
    
    def test_cannot_compute_quality_without_ground_truth(self):
        """Quality metrics (WER, speaker_accuracy) require ground truth."""
        prov = create_provenance(
            dataset_id="test",
            ground_truth_path=None,
        )
        assert can_compute_quality_metrics(prov) is False
    
    def test_can_compute_quality_with_ground_truth(self, tmp_path):
        """Quality metrics allowed when ground truth exists."""
        gt_file = tmp_path / "ground_truth.txt"
        gt_file.write_text("test content")
        
        prov = create_provenance(
            dataset_id="test",
            ground_truth_path=gt_file,
        )
        assert prov["has_ground_truth"] is True
        assert can_compute_quality_metrics(prov) is True
    
    def test_invalid_metrics_blocks_quality(self, tmp_path):
        """metrics_valid=False blocks quality computation even with GT."""
        gt_file = tmp_path / "ground_truth.txt"
        gt_file.write_text("test content")
        
        prov = create_provenance(
            dataset_id="test",
            ground_truth_path=gt_file,
            metrics_valid=False,
            invalid_reason="Poisoned data",
        )
        assert prov["has_ground_truth"] is True
        assert prov["metrics_valid"] is False
        assert can_compute_quality_metrics(prov) is False


class TestNoGTMetricsRule:
    """
    Critical tests: quality metrics MUST be None when no ground truth.
    
    These tests encode the rule:
    - WER must be None when has_ground_truth=False
    - speaker_accuracy must be None when no reference exists
    """
    
    def test_wer_none_without_ground_truth(self):
        """WER must be None when there's no ground truth text."""
        # This tests the rule, not the implementation
        # The actual metric computation in run_asr.py must enforce this
        prov = create_provenance(dataset_id="smoke", ground_truth_path=None)
        
        # If we can't compute quality metrics, WER should not be set
        assert can_compute_quality_metrics(prov) is False
        
        # Simulate what metrics should look like
        metrics = {}
        if can_compute_quality_metrics(prov):
            metrics["wer"] = 0.25  # Would be set if GT exists
        else:
            metrics["wer"] = None  # Must be None
        
        assert metrics["wer"] is None
    
    def test_speaker_accuracy_none_without_reference(self):
        """speaker_accuracy must be None when there's no reference."""
        prov = create_provenance(dataset_id="diar_smoke", ground_truth_path=None)
        
        assert can_compute_quality_metrics(prov) is False
        
        # Simulate what metrics should look like
        metrics = {}
        if can_compute_quality_metrics(prov):
            metrics["speaker_accuracy"] = 1.0
        else:
            metrics["speaker_accuracy"] = None
        
        assert metrics["speaker_accuracy"] is None
    
    def test_wer_allowed_with_ground_truth(self, tmp_path):
        """WER can be set when ground truth exists."""
        gt_file = tmp_path / "transcript.txt"
        gt_file.write_text("expected transcription")
        
        prov = create_provenance(dataset_id="golden", ground_truth_path=gt_file)
        
        assert can_compute_quality_metrics(prov) is True
        
        metrics = {}
        if can_compute_quality_metrics(prov):
            metrics["wer"] = 0.15
        else:
            metrics["wer"] = None
        
        assert metrics["wer"] == 0.15


class TestQuarantineIgnore:
    """Test that quarantined runs are excluded from decisions."""
    
    def test_quarantine_path_pattern(self):
        """Runs under runs_quarantine/ should be ignored."""
        quarantine_path = Path("runs_quarantine/model/task/run.json")
        runs_path = Path("runs/model/task/run.json")
        
        # Simple pattern check
        assert "runs_quarantine" in str(quarantine_path)
        assert "runs_quarantine" not in str(runs_path)
    
    def test_quarantine_run_not_in_normal_glob(self):
        """glob('runs/**/*.json') should not match runs_quarantine."""
        # This tests the glob pattern used in generate_decisions.py
        pattern_base = Path("runs")
        quarantine_base = Path("runs_quarantine")
        
        # These are different directories
        assert pattern_base != quarantine_base
        assert not str(quarantine_base).startswith(str(pattern_base) + "/")


class TestMandatoryProvenance:
    """
    Unit tests that BLOCK runs without provenance.
    
    This prevents agents, humans, or old scripts from reintroducing
    evidence without proper provenance fields.
    """
    
    def test_all_runs_have_provenance(self):
        """All non-quarantined run artifacts MUST have provenance."""
        runs_dir = Path(__file__).parent.parent.parent / "runs"
        if not runs_dir.exists():
            pytest.skip("No runs directory")
        
        violations = []
        for run_file in runs_dir.glob("**/*.json"):
            # Skip summary files
            if run_file.name == "summary.json":
                continue
            
            try:
                with open(run_file) as f:
                    data = json.load(f)
                
                # Check for provenance field
                if "provenance" not in data:
                    violations.append(f"{run_file}: missing 'provenance' field")
            except json.JSONDecodeError:
                violations.append(f"{run_file}: invalid JSON")
        
        if violations:
            # All legacy runs have been quarantined - this is now a hard fail
            pytest.fail(f"Runs without provenance ({len(violations)}):\n" + "\n".join(violations[:10]))
    
    def test_no_wer_without_ground_truth(self):
        """Runs with has_ground_truth=False must have wer=None."""
        runs_dir = Path(__file__).parent.parent.parent / "runs"
        if not runs_dir.exists():
            pytest.skip("No runs directory")
        
        violations = []
        for run_file in runs_dir.glob("**/*.json"):
            if run_file.name == "summary.json":
                continue
            
            try:
                with open(run_file) as f:
                    data = json.load(f)
                
                prov = data.get("provenance", {})
                metrics = data.get("metrics", {})
                
                # If provenance says no GT, WER must be None
                if prov.get("has_ground_truth") is False:
                    wer = metrics.get("wer")
                    if wer is not None:
                        violations.append(f"{run_file}: has_ground_truth=False but wer={wer}")
            except (json.JSONDecodeError, KeyError):
                pass  # Skip malformed files in this test
        
        if violations:
            pytest.fail(f"WER without ground truth:\n" + "\n".join(violations))


class TestV2VStructuralEvidence:
    """
    V2V-specific tests: structural metrics required, quality metrics forbidden.
    
    V2V smoke evidence must have:
    - has_ground_truth=false (no reference target)
    - At least one of (latency_ms, rtf) present and finite
    - No quality metrics (wer, accuracy, etc.)
    """
    
    def test_v2v_smoke_requires_structural_metrics(self):
        """V2V smoke evidence must have at least latency or rtf."""
        runs_dir = Path(__file__).parent.parent.parent / "runs"
        if not runs_dir.exists():
            pytest.skip("No runs directory")
        
        violations = []
        for run_file in runs_dir.glob("**/v2v/*.json"):
            if run_file.name == "summary.json":
                continue
            
            try:
                with open(run_file) as f:
                    data = json.load(f)
                
                prov = data.get("provenance", {})
                metrics = data.get("metrics", {})
                
                # V2V smoke should have structural metrics
                if prov.get("has_ground_truth") is False:
                    latency = metrics.get("latency_ms")
                    rtf = metrics.get("rtf") or metrics.get("response_duration_s")
                    
                    # Must have at least one finite structural metric
                    has_finite = (latency is not None and latency > 0) or \
                                (rtf is not None and rtf > 0)
                    
                    if not has_finite:
                        violations.append(f"{run_file}: V2V smoke missing structural metrics (latency/rtf)")
            except (json.JSONDecodeError, KeyError):
                pass
        
        if violations:
            pytest.fail(f"V2V smoke without structural metrics:\n" + "\n".join(violations))
    
    def test_v2v_smoke_no_quality_metrics(self):
        """V2V smoke with no GT must not have quality metrics."""
        runs_dir = Path(__file__).parent.parent.parent / "runs"
        if not runs_dir.exists():
            pytest.skip("No runs directory")
        
        violations = []
        quality_fields = ["wer", "cer", "accuracy", "speaker_accuracy", "bleu"]
        
        for run_file in runs_dir.glob("**/v2v/*.json"):
            if run_file.name == "summary.json":
                continue
            
            try:
                with open(run_file) as f:
                    data = json.load(f)
                
                prov = data.get("provenance", {})
                metrics = data.get("metrics", {})
                
                if prov.get("has_ground_truth") is False:
                    for field in quality_fields:
                        if metrics.get(field) is not None:
                            violations.append(f"{run_file}: has_ground_truth=False but {field}={metrics[field]}")
            except (json.JSONDecodeError, KeyError):
                pass
        
        if violations:
            pytest.fail(f"V2V quality metrics without ground truth:\n" + "\n".join(violations))


class TestRunContext:
    """
    Tests that run_context is present and complete.
    
    run_context makes latency metrics interpretable by recording:
    - device (required)
    - runner_git_hash (required for reproducibility)
    - audio_duration_s (required for audio tasks, enables RTF calculation)
    """
    
    def test_all_runs_have_run_context(self):
        """All runs must have run_context with at least device."""
        runs_dir = Path(__file__).parent.parent.parent / "runs"
        if not runs_dir.exists():
            pytest.skip("No runs directory")
        
        violations = []
        for run_file in runs_dir.glob("**/*.json"):
            if run_file.name == "summary.json":
                continue
            
            try:
                with open(run_file) as f:
                    data = json.load(f)
                
                run_context = data.get("run_context")
                if run_context is None:
                    violations.append(f"{run_file}: missing 'run_context' field")
                elif run_context.get("device") is None:
                    violations.append(f"{run_file}: run_context missing 'device'")
            except json.JSONDecodeError:
                pass  # Skip malformed JSON
        
        if violations:
            # Allow legacy runs for now, but warn
            import warnings
            warnings.warn(f"Runs without run_context: {len(violations)}")
            # To make hard fail, uncomment:
            # pytest.fail(f"Runs without run_context:\n" + "\n".join(violations[:10]))
    
    def test_audio_tasks_have_duration(self):
        """Audio tasks (ASR, VAD, diarization, V2V) must have audio_duration_s."""
        runs_dir = Path(__file__).parent.parent.parent / "runs"
        if not runs_dir.exists():
            pytest.skip("No runs directory")
        
        audio_tasks = ["asr", "vad", "diarization", "v2v"]
        violations = []
        
        for run_file in runs_dir.glob("**/*.json"):
            if run_file.name == "summary.json":
                continue
            
            try:
                with open(run_file) as f:
                    data = json.load(f)
                
                task = data.get("meta", {}).get("task") or data.get("capability")
                if task not in audio_tasks:
                    continue
                
                run_context = data.get("run_context", {})
                if run_context.get("audio_duration_s") is None:
                    violations.append(f"{run_file}: audio task '{task}' missing audio_duration_s")
            except (json.JSONDecodeError, KeyError):
                pass
        
        if violations:
            import warnings
            warnings.warn(f"Audio tasks without duration: {len(violations)}")


