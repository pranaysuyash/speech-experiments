"""
Tests for enhancement metrics (LCS-03).

CI-safe, synthetic signal tests. No real audio required.
"""

import numpy as np
import pytest

from harness.metrics_enhance import (
    si_snr,
    stoi,
    pesq,
    is_stoi_available,
    is_pesq_available,
)


class TestSiSnr:
    """Tests for SI-SNR metric."""

    def test_identical_signals_high_value(self):
        """Identical signals should give very high SI-SNR."""
        ref = np.random.randn(16000)
        result = si_snr(ref, ref)
        # Perfect reconstruction gives inf
        assert result == float("inf")

    def test_scaled_signal_invariance(self):
        """SI-SNR should be scale invariant."""
        ref = np.random.randn(16000)
        est = ref * 0.5  # Scaled version
        
        result = si_snr(ref, est)
        # Scaled version of same signal still gives inf (perfect)
        assert result == float("inf")

    def test_inverted_signal_is_scale_invariant(self):
        """SI-SNR is scale invariant: -x is considered same as x."""
        ref = np.random.randn(16000)
        est = -ref  # Inverted = scaled by -1
        
        result = si_snr(ref, est)
        # Scale invariance means inverted signal still gives inf
        assert result == float("inf")

    def test_uncorrelated_noise_low_value(self):
        """Uncorrelated noise should give low SI-SNR."""
        np.random.seed(42)
        ref = np.random.randn(16000)
        np.random.seed(123)  # Different seed for uncorrelated
        est = np.random.randn(16000)
        
        result = si_snr(ref, est)
        # Uncorrelated noise gives low SI-SNR
        assert result < 10  # Much less than inf

    def test_noise_gives_finite_value(self):
        """Adding noise should give finite positive SI-SNR."""
        ref = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)
        noise = np.random.randn(16000) * 0.1
        est = ref + noise
        
        result = si_snr(ref, est)
        assert np.isfinite(result)
        assert result > 0  # Still mostly correct

    def test_length_mismatch_raises(self):
        """Length mismatch should raise ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            si_snr(np.zeros(100), np.zeros(200))

    def test_empty_raises(self):
        """Empty arrays should raise ValueError."""
        with pytest.raises(ValueError, match="Empty"):
            si_snr(np.array([]), np.array([]))

    def test_silent_reference_raises(self):
        """Silent reference should raise ValueError."""
        with pytest.raises(ValueError, match="silent"):
            si_snr(np.zeros(100), np.random.randn(100))


class TestStoi:
    """Tests for STOI metric."""

    def test_skipped_if_not_available(self):
        """STOI returns skip info if pystoi not installed."""
        if is_stoi_available():
            pytest.skip("pystoi is installed, skip test not applicable")
        
        ref = np.random.randn(16000)
        result = stoi(ref, ref, sr=16000)
        
        assert result["skipped"] is True
        assert "pystoi" in result["reason"]

    def test_available_returns_score(self):
        """STOI returns score if pystoi is installed."""
        if not is_stoi_available():
            pytest.skip("pystoi not installed")
        
        ref = np.random.randn(16000)
        result = stoi(ref, ref, sr=16000)
        
        assert result["skipped"] is False
        assert "score" in result
        assert 0 <= result["score"] <= 1


class TestPesq:
    """Tests for PESQ metric."""

    def test_skipped_if_not_available(self):
        """PESQ returns skip info if pesq not installed."""
        if is_pesq_available():
            pytest.skip("pesq is installed, skip test not applicable")
        
        ref = np.random.randn(16000)
        result = pesq(ref, ref, sr=16000)
        
        assert result["skipped"] is True
        assert "pesq" in result["reason"].lower()

    def test_never_fails_ci(self):
        """PESQ must never raise - always returns dict."""
        ref = np.random.randn(16000)
        
        # Should not raise, regardless of dependency
        result = pesq(ref, ref, sr=16000)
        
        assert isinstance(result, dict)
        assert "skipped" in result
