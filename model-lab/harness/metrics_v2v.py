"""
V2V evaluation metrics.
Focuses on latency, audio validity, and turn completion.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class V2VResult:
    """Container for V2V evaluation results."""

    latency_ms: float
    response_duration_s: float
    has_audio: bool
    metadata: dict[str, Any]


class V2VMetrics:
    """Calculate V2V evaluation metrics."""

    @staticmethod
    def evaluate(
        response_audio: np.ndarray | None,
        sr: int,
        latency_s: float,
        metadata: dict[str, Any] | None = None,
    ) -> V2VResult:
        """
        Evaluate V2V response.

        Args:
            response_audio: Output audio array (frame count,)
            sr: Sample rate
            latency_s: Time to first audio (or full generation time if batch)
        """
        has_audio = response_audio is not None and len(response_audio) > 0

        if has_audio:
            duration = len(response_audio) / sr
        else:
            duration = 0.0

        latency_ms = latency_s * 1000

        result = V2VResult(
            latency_ms=latency_ms,
            response_duration_s=duration,
            has_audio=has_audio,
            metadata=metadata or {},
        )

        logger.info(f"V2V Eval: Latency={latency_ms:.1f}ms, Audio={duration:.1f}s")
        return result
