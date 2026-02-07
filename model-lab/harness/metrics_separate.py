"""
Separation metrics for source separation models.

LCS-03: CI-safe metrics for separate surface.
- SDR/SIR/SAR: requires mir_eval (optional)
- All metrics work on arrays, no dataset coupling
"""

from __future__ import annotations

import numpy as np
from typing import Any

# Track optional dependency
_MIR_EVAL_AVAILABLE = False

try:
    import mir_eval.separation as _mir_sep
    _MIR_EVAL_AVAILABLE = True
except ImportError:
    _mir_sep = None


def bss_eval(
    reference_sources: np.ndarray,
    estimated_sources: np.ndarray,
) -> dict[str, Any]:
    """
    Compute BSS Eval metrics (SDR, SIR, SAR, permutation).
    
    Uses mir_eval if available. Returns skip info otherwise.
    
    Args:
        reference_sources: Shape (n_sources, n_samples)
        estimated_sources: Shape (n_sources, n_samples)
    
    Returns:
        Dict with 'sdr', 'sir', 'sar' arrays, 'perm' array, or 'skipped'
    """
    if not _MIR_EVAL_AVAILABLE:
        return {"skipped": True, "reason": "mir_eval not installed"}
    
    reference_sources = np.asarray(reference_sources, dtype=np.float64)
    estimated_sources = np.asarray(estimated_sources, dtype=np.float64)
    
    if reference_sources.shape != estimated_sources.shape:
        raise ValueError(
            f"Shape mismatch: reference={reference_sources.shape}, "
            f"estimated={estimated_sources.shape}"
        )
    
    if reference_sources.ndim == 1:
        # Single source: reshape to (1, n_samples)
        reference_sources = reference_sources.reshape(1, -1)
        estimated_sources = estimated_sources.reshape(1, -1)
    
    sdr, sir, sar, perm = _mir_sep.bss_eval_sources(
        reference_sources, estimated_sources
    )
    
    return {
        "sdr": sdr.tolist(),
        "sir": sir.tolist(),
        "sar": sar.tolist(),
        "perm": perm.tolist(),
        "skipped": False,
    }


def sdr(reference: np.ndarray, estimate: np.ndarray) -> dict[str, Any]:
    """
    Compute Signal-to-Distortion Ratio for a single source.
    
    Convenience wrapper around bss_eval for single-source case.
    
    Args:
        reference: Clean reference signal (1D)
        estimate: Estimated signal (1D)
    
    Returns:
        Dict with 'sdr' float, or 'skipped'
    """
    result = bss_eval(
        np.asarray(reference).reshape(1, -1),
        np.asarray(estimate).reshape(1, -1),
    )
    
    if result.get("skipped"):
        return result
    
    return {"sdr": result["sdr"][0], "skipped": False}


def multi_source_sdr(
    reference_sources: np.ndarray,
    estimated_sources: np.ndarray,
) -> dict[str, Any]:
    """
    Compute per-source SDR for multi-source separation.
    
    Args:
        reference_sources: Shape (n_sources, n_samples)
        estimated_sources: Shape (n_sources, n_samples)
    
    Returns:
        Dict with 'sdr' (per-source list), 'mean_sdr', or 'skipped'
    """
    result = bss_eval(reference_sources, estimated_sources)
    
    if result.get("skipped"):
        return result
    
    sdr_values = result["sdr"]
    return {
        "sdr": sdr_values,
        "mean_sdr": float(np.mean(sdr_values)),
        "skipped": False,
    }


def is_mir_eval_available() -> bool:
    """Check if mir_eval dependency is available."""
    return _MIR_EVAL_AVAILABLE
