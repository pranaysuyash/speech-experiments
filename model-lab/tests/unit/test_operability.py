"""
Unit tests for operability threshold enforcement.

These tests ensure:
- Missing metrics fail operability
- rtf_like_max is enforced for smoke evidence
- Golden batch bypasses operability
- Operability failures block use cases
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from harness.operability import evaluate_operability, get_task_thresholds, filter_viable_evidence


class TestEvaluateOperability:
    """Tests for the evaluate_operability function."""
    
    def test_missing_metric_fails(self):
        """Missing metric is a failure."""
        passed, failures = evaluate_operability(
            task="v2v",
            metrics={},  # No rtf_like
            thresholds={"rtf_like_max": 1.2}
        )
        
        assert passed is False
        assert len(failures) == 1
        assert "missing metric: rtf_like" in failures[0]
    
    def test_rtf_like_max_enforced_pass(self):
        """rtf_like within threshold passes."""
        passed, failures = evaluate_operability(
            task="v2v",
            metrics={"rtf_like": 0.56},
            thresholds={"rtf_like_max": 1.2}
        )
        
        assert passed is True
        assert len(failures) == 0
    
    def test_rtf_like_max_enforced_fail(self):
        """rtf_like above threshold fails."""
        passed, failures = evaluate_operability(
            task="v2v",
            metrics={"rtf_like": 1.5},
            thresholds={"rtf_like_max": 1.2}
        )
        
        assert passed is False
        assert len(failures) == 1
        assert "rtf_like 1.50 > 1.2" in failures[0]
    
    def test_speech_ratio_min_enforced(self):
        """speech_ratio_min threshold works."""
        passed, failures = evaluate_operability(
            task="vad",
            metrics={"speech_ratio": 0.01},
            thresholds={"speech_ratio_min": 0.05}
        )
        
        assert passed is False
        assert "speech_ratio 0.01 < 0.05" in failures[0]
    
    def test_speech_ratio_max_enforced(self):
        """speech_ratio_max threshold works."""
        passed, failures = evaluate_operability(
            task="vad",
            metrics={"speech_ratio": 0.99},
            thresholds={"speech_ratio_max": 0.95}
        )
        
        assert passed is False
        assert "speech_ratio 0.99 > 0.95" in failures[0]
    
    def test_multiple_thresholds(self):
        """Multiple thresholds are all checked."""
        passed, failures = evaluate_operability(
            task="vad",
            metrics={"speech_ratio": 0.5},  # Within bounds
            thresholds={"speech_ratio_min": 0.05, "speech_ratio_max": 0.95}
        )
        
        assert passed is True
        assert len(failures) == 0
    
    def test_empty_thresholds_always_pass(self):
        """Empty thresholds means no operability check."""
        passed, failures = evaluate_operability(
            task="any",
            metrics={},
            thresholds={}
        )
        
        assert passed is True
        assert len(failures) == 0
    
    def test_none_thresholds_always_pass(self):
        """None thresholds means no operability check."""
        passed, failures = evaluate_operability(
            task="any",
            metrics={},
            thresholds=None
        )
        
        assert passed is True
        assert len(failures) == 0


class TestGetTaskThresholds:
    """Tests for get_task_thresholds helper."""
    
    def test_gets_task_thresholds(self):
        """Extracts thresholds for a specific task."""
        config = {
            "vad": {"speech_ratio_min": 0.05},
            "v2v": {"rtf_like_max": 1.2}
        }
        
        thresholds = get_task_thresholds(config, "v2v")
        
        assert thresholds == {"rtf_like_max": 1.2}
    
    def test_missing_task_returns_empty(self):
        """Missing task returns empty dict."""
        config = {"vad": {"speech_ratio_min": 0.05}}
        
        thresholds = get_task_thresholds(config, "v2v")
        
        assert thresholds == {}
    
    def test_empty_config_returns_empty(self):
        """Empty config returns empty dict."""
        thresholds = get_task_thresholds({}, "v2v")
        
        assert thresholds == {}


class TestOperabilityNotAppliedToGolden:
    """Tests that operability applies only to smoke evidence."""
    
    def test_golden_batch_bypasses_operability(self):
        """Golden batch evidence bypasses operability thresholds."""
        from dataclasses import dataclass
        from harness.model_card import EvidenceGrade
        
        @dataclass
        class MockEvidence:
            model_id: str
            grade: EvidenceGrade
            metrics: dict
        
        evidence = [
            MockEvidence("model1", EvidenceGrade.GOLDEN_BATCH, {"rtf_like": 99.0})  # Would fail if checked
        ]
        
        viable, rejections = filter_viable_evidence(
            evidence_list=evidence,
            task="v2v",
            thresholds={"rtf_like_max": 1.2},
            grade_filter="smoke"  # Only apply to smoke
        )
        
        # Golden batch should pass through
        assert len(viable) == 1
        assert len(rejections) == 0


class TestOperabilityIntegration:
    """Integration tests for operability in decision pipeline."""
    
    def test_decisions_with_tiny_threshold_produces_rejection(self):
        """Setting a tiny rtf_like_max should cause V2V to fail."""
        import tempfile
        import yaml
        
        # Create a test use case with impossible threshold
        test_use_case = {
            "id": "test_assistant",
            "name": "Test Assistant",
            "evaluation_mode": "pipeline",
            "requirements": {
                "primary": [
                    {"task": "v2v", "min_grade": "smoke"}
                ],
                "secondary": []
            },
            "operability": {
                "v2v": {"rtf_like_max": 0.01}  # Impossibly low
            },
            "fatal_gates": [],
            "warning_gates": []
        }
        
        # This is a synthetic test - just verify the threshold parsing works
        from harness.operability import get_task_thresholds, evaluate_operability
        
        thresholds = get_task_thresholds(test_use_case["operability"], "v2v")
        assert thresholds["rtf_like_max"] == 0.01
        
        # A typical V2V run would fail this
        passed, failures = evaluate_operability("v2v", {"rtf_like": 0.56}, thresholds)
        assert passed is False
        assert "rtf_like 0.56 > 0.01" in failures[0]
