"""
Enhancement metrics for audio enhancement models.

LCS-03: CI-safe metrics for enhance surface.
- SI-SNR: always available (pure numpy)
- STOI: optional (requires pystoi)
- PESQ: optional, never fails CI (requires pesq)
"""

from __future__ import annotations

import numpy as np
from typing import Any

# Track optional dependencies
_PYSTOI_AVAILABLE = False
_PESQ_AVAILABLE = False

try:
    from pystoi import stoi as _stoi_impl
    _PYSTOI_AVAILABLE = True
except ImportError:
    _stoi_impl = None

try:
    from pesq import pesq as _pesq_impl
    _PESQ_AVAILABLE = True
except ImportError:
    _pesq_impl = None


def si_snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """
    Compute Scale-Invariant Signal-to-Noise Ratio (SI-SNR).
    
    SI-SNR is scale invariant: scaling the estimate does not change the result.
    
    Args:
        reference: Clean reference signal (1D array)
        estimate: Enhanced/estimated signal (1D array)
    
    Returns:
        SI-SNR in dB (higher is better)
    
    Raises:
        ValueError: If shapes don't match or inputs are invalid
    """
    reference = np.asarray(reference, dtype=np.float64).flatten()
    estimate = np.asarray(estimate, dtype=np.float64).flatten()
    
    if len(reference) != len(estimate):
        raise ValueError(f"Length mismatch: reference={len(reference)}, estimate={len(estimate)}")
    
    if len(reference) == 0:
        raise ValueError("Empty arrays")
    
    # Zero-mean normalization
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    
    # Compute target: projection of estimate onto reference
    ref_energy = np.dot(reference, reference)
    if ref_energy < 1e-10:
        raise ValueError("Reference signal is silent (zero energy)")
    
    # s_target = <s', s> / ||s||^2 * s
    s_target = np.dot(estimate, reference) / ref_energy * reference
    
    # e_noise = s' - s_target
    e_noise = estimate - s_target
    
    target_energy = np.dot(s_target, s_target)
    noise_energy = np.dot(e_noise, e_noise)
    
    if noise_energy < 1e-10:
        # Perfect reconstruction
        return float("inf")
    
    si_snr_value = 10 * np.log10(target_energy / noise_energy)
    return float(si_snr_value)


def stoi(
    reference: np.ndarray,
    estimate: np.ndarray,
    sr: int,
    extended: bool = False,
) -> dict[str, Any]:
    """
    Compute Short-Time Objective Intelligibility (STOI).
    
    Requires pystoi package. Returns skip info if not available.
    
    Args:
        reference: Clean reference signal
        estimate: Enhanced signal  
        sr: Sample rate (typically 16000)
        extended: Use extended STOI (default False)
    
    Returns:
        Dict with 'score' in [0, 1], or 'skipped' with reason
    """
    if not _PYSTOI_AVAILABLE:
        return {"skipped": True, "reason": "pystoi not installed"}
    
    reference = np.asarray(reference, dtype=np.float64).flatten()
    estimate = np.asarray(estimate, dtype=np.float64).flatten()
    
    if len(reference) != len(estimate):
        raise ValueError(f"Length mismatch: reference={len(reference)}, estimate={len(estimate)}")
    
    score = _stoi_impl(reference, estimate, sr, extended=extended)
    return {"score": float(score), "skipped": False}


def pesq(
    reference: np.ndarray,
    estimate: np.ndarray,
    sr: int,
    mode: str = "wb",
) -> dict[str, Any]:
    """
    Compute Perceptual Evaluation of Speech Quality (PESQ).
    
    OPTIONAL metric - never fails CI. Returns skip info if not available.
    
    Args:
        reference: Clean reference signal
        estimate: Enhanced signal
        sr: Sample rate (8000 for nb, 16000 for wb)
        mode: 'nb' (narrowband) or 'wb' (wideband)
    
    Returns:
        Dict with 'score' (MOS-LQO), or 'skipped' with reason
    """
    if not _PESQ_AVAILABLE:
        return {"skipped": True, "reason": "pesq not installed (optional, CI-safe to skip)"}
    
    reference = np.asarray(reference, dtype=np.float64).flatten()
    estimate = np.asarray(estimate, dtype=np.float64).flatten()
    
    if len(reference) != len(estimate):
        raise ValueError(f"Length mismatch: reference={len(reference)}, estimate={len(estimate)}")
    
    try:
        score = _pesq_impl(sr, reference, estimate, mode)
        return {"score": float(score), "skipped": False}
    except Exception as e:
        return {"skipped": True, "reason": f"pesq error: {e}"}


def is_stoi_available() -> bool:
    """Check if STOI dependency is available."""
    return _PYSTOI_AVAILABLE


def is_pesq_available() -> bool:
    """Check if PESQ dependency is available."""
    return _PESQ_AVAILABLE
