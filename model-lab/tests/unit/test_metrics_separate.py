"""
Tests for separation metrics (LCS-03).

CI-safe, synthetic signal tests. No MUSDB required.
"""

import numpy as np
import pytest

from harness.metrics_separate import (
    bss_eval,
    sdr,
    multi_source_sdr,
    is_mir_eval_available,
)


class TestBssEval:
    """Tests for BSS Eval metrics."""

    def test_skipped_if_not_available(self):
        """BSS Eval returns skip info if mir_eval not installed."""
        if is_mir_eval_available():
            pytest.skip("mir_eval is installed, skip test not applicable")
        
        ref = np.random.randn(2, 16000)
        result = bss_eval(ref, ref)
        
        assert result["skipped"] is True
        assert "mir_eval" in result["reason"]

    def test_available_returns_metrics(self):
        """BSS Eval returns SDR/SIR/SAR if mir_eval installed."""
        if not is_mir_eval_available():
            pytest.skip("mir_eval not installed")
        
        ref = np.random.randn(2, 16000)
        result = bss_eval(ref, ref)
        
        assert result["skipped"] is False
        assert "sdr" in result
        assert "sir" in result
        assert "sar" in result
        assert len(result["sdr"]) == 2

    def test_shape_mismatch_raises(self):
        """Shape mismatch should raise ValueError."""
        if not is_mir_eval_available():
            pytest.skip("mir_eval not installed")
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            bss_eval(np.zeros((2, 100)), np.zeros((3, 100)))


class TestSdr:
    """Tests for single-source SDR."""

    def test_skipped_if_not_available(self):
        """SDR returns skip info if mir_eval not installed."""
        if is_mir_eval_available():
            pytest.skip("mir_eval is installed")
        
        ref = np.random.randn(16000)
        result = sdr(ref, ref)
        
        assert result["skipped"] is True

    def test_perfect_reconstruction(self):
        """Perfect reconstruction should give high SDR."""
        if not is_mir_eval_available():
            pytest.skip("mir_eval not installed")
        
        ref = np.random.randn(16000)
        result = sdr(ref, ref)
        
        assert result["skipped"] is False
        assert result["sdr"] > 100  # Very high for perfect


class TestMultiSourceSdr:
    """Tests for multi-source SDR."""

    def test_trivial_mixture(self):
        """Stem equals mixture, others zero: should be finite."""
        if not is_mir_eval_available():
            pytest.skip("mir_eval not installed")
        
        # Two sources: one is the signal, one is zeros + tiny noise
        signal = np.random.randn(16000)
        ref = np.array([signal, np.zeros(16000) + 1e-10 * np.random.randn(16000)])
        est = np.array([signal, np.zeros(16000) + 1e-10 * np.random.randn(16000)])
        
        result = multi_source_sdr(ref, est)
        
        assert result["skipped"] is False
        assert np.isfinite(result["mean_sdr"])

    def test_returns_mean_sdr(self):
        """Should return per-source SDR and mean."""
        if not is_mir_eval_available():
            pytest.skip("mir_eval not installed")
        
        ref = np.random.randn(2, 16000)
        result = multi_source_sdr(ref, ref)
        
        assert "sdr" in result
        assert "mean_sdr" in result
        assert len(result["sdr"]) == 2


class TestDependencyHandling:
    """Tests for clean dependency skip behavior."""

    def test_all_functions_return_dict(self):
        """All metric functions must return dict, never raise on missing deps."""
        ref = np.random.randn(16000)
        
        # These should never raise
        result1 = bss_eval(ref.reshape(1, -1), ref.reshape(1, -1))
        result2 = sdr(ref, ref)
        result3 = multi_source_sdr(ref.reshape(1, -1), ref.reshape(1, -1))
        
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)
        assert isinstance(result3, dict)
