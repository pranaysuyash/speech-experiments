"""
ASR evaluation metrics for model comparison.
Implements WER, CER, and related metrics with proper error tracking.
"""

import numpy as np
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ASRResult:
    """Container for ASR evaluation results."""
    text: str
    ground_truth: str
    wer: float
    cer: float
    latency_ms: float
    rtv: float  # Real-Time Factor (processing time / audio duration)
    metadata: Dict[str, Any]


class ASRMetrics:
    """Calculate ASR evaluation metrics."""

    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> Tuple[float, int, int, int]:
        """
        Calculate Word Error Rate (WER).

        WER = (substitutions + deletions + insertions) / total_reference_words

        Returns:
            wer: Word error rate (0-1)
            substitutions: Number of substitutions
            deletions: Number of deletions
            insertions: Number of insertions
        """
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()

        # Dynamic programming for edit distance
        dp = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))

        for i in range(len(ref_words) + 1):
            dp[i, 0] = i
        for j in range(len(hyp_words) + 1):
            dp[0, j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    dp[i, j] = dp[i-1, j-1]
                else:
                    dp[i, j] = min(
                        dp[i-1, j] + 1,    # deletion
                        dp[i, j-1] + 1,    # insertion
                        dp[i-1, j-1] + 1   # substitution
                    )

        # Backtrack to count error types
        i, j = len(ref_words), len(hyp_words)
        substitutions = deletions = insertions = 0

        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref_words[i-1] == hyp_words[j-1]:
                i, j = i-1, j-1
            elif i > 0 and j > 0 and dp[i, j] == dp[i-1, j-1] + 1:
                substitutions += 1
                i, j = i-1, j-1
            elif i > 0 and dp[i, j] == dp[i-1, j] + 1:
                deletions += 1
                i -= 1
            else:
                insertions += 1
                j -= 1

        wer = dp[len(ref_words), len(hyp_words)] / max(1, len(ref_words))
        return wer, substitutions, deletions, insertions

    @staticmethod
    def calculate_cer(reference: str, hypothesis: str) -> float:
        """
        Calculate Character Error Rate (CER).

        CER = (substitutions + deletions + insertions) / total_reference_chars
        """
        ref_chars = list(reference.lower())
        hyp_chars = list(hypothesis.lower())

        # Levenshtein distance
        dp = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1))

        for i in range(len(ref_chars) + 1):
            dp[i, 0] = i
        for j in range(len(hyp_chars) + 1):
            dp[0, j] = j

        for i in range(1, len(ref_chars) + 1):
            for j in range(1, len(hyp_chars) + 1):
                if ref_chars[i-1] == hyp_chars[j-1]:
                    dp[i, j] = dp[i-1, j-1]
                else:
                    dp[i, j] = min(
                        dp[i-1, j] + 1,    # deletion
                        dp[i, j-1] + 1,    # insertion
                        dp[i-1, j-1] + 1   # substitution
                    )

        cer = dp[len(ref_chars), len(hyp_chars)] / max(1, len(ref_chars))
        return cer

    @staticmethod
    def evaluate(transcription: str,
                 ground_truth: str,
                 audio_duration_s: float,
                 latency_s: float,
                 metadata: Dict[str, Any] = None) -> ASRResult:
        """
        Comprehensive ASR evaluation.

        Args:
            transcription: Model output text
            ground_truth: Reference text
            audio_duration_s: Audio length in seconds
            latency_s: Processing time in seconds
            metadata: Additional metadata

        Returns:
            ASRResult with all metrics
        """
        wer, sub, del_, ins = ASRMetrics.calculate_wer(ground_truth, transcription)
        cer = ASRMetrics.calculate_cer(ground_truth, transcription)

        rtv = latency_s / max(0.001, audio_duration_s)  # Real-Time Factor
        latency_ms = latency_s * 1000

        result = ASRResult(
            text=transcription,
            ground_truth=ground_truth,
            wer=wer,
            cer=cer,
            latency_ms=latency_ms,
            rtv=rtv,
            metadata=metadata or {}
        )

        # Log breakdown
        logger.info(f"ASR Evaluation:")
        logger.info(f"  WER: {wer:.3f} (S:{sub}, D:{del_}, I:{ins})")
        logger.info(f"  CER: {cer:.3f}")
        logger.info(f"  Latency: {latency_ms:.1f}ms")
        logger.info(f"  RTF: {rtv:.3f}x")

        return result


class ASRBatcher:
    """Batch evaluation for multiple tests."""

    @staticmethod
    def evaluate_batch(results: List[ASRResult]) -> Dict[str, Any]:
        """
        Calculate aggregate metrics across multiple ASR results.

        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        if not results:
            return {}

        metrics = {
            'wer_mean': np.mean([r.wer for r in results]),
            'wer_std': np.std([r.wer for r in results]),
            'cer_mean': np.mean([r.cer for r in results]),
            'cer_std': np.std([r.cer for r in results]),
            'latency_ms_mean': np.mean([r.latency_ms for r in results]),
            'latency_ms_std': np.std([r.latency_ms for r in results]),
            'rtv_mean': np.mean([r.rtv for r in results]),
            'rtv_std': np.std([r.rtv for r in results]),
            'num_samples': len(results)
        }

        logger.info(f"Batch evaluation ({len(results)} samples):")
        logger.info(f"  Mean WER: {metrics['wer_mean']:.3f} ± {metrics['wer_std']:.3f}")
        logger.info(f"  Mean CER: {metrics['cer_mean']:.3f} ± {metrics['cer_std']:.3f}")
        logger.info(f"  Mean RTF: {metrics['rtv_mean']:.3f} ± {metrics['rtv_std']:.3f}")

        return metrics


def format_asr_result(result: ASRResult) -> str:
    """Format ASR result for logging."""
    return (
        f"WER: {result.wer:.3f}, "
        f"CER: {result.cer:.3f}, "
        f"Latency: {result.latency_ms:.1f}ms, "
        f"RTF: {result.rtv:.3f}x"
    )