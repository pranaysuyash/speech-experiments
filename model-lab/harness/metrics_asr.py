"""
ASR evaluation metrics for model comparison.
Implements WER, CER, and related metrics with proper error tracking.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

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
    metadata: dict[str, Any]


class ASRMetrics:
    """Calculate ASR evaluation metrics."""

    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> tuple[float, int, int, int]:
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
                if ref_words[i - 1] == hyp_words[j - 1]:
                    dp[i, j] = dp[i - 1, j - 1]
                else:
                    dp[i, j] = min(
                        dp[i - 1, j] + 1,  # deletion
                        dp[i, j - 1] + 1,  # insertion
                        dp[i - 1, j - 1] + 1,  # substitution
                    )

        # Backtrack to count error types
        i, j = len(ref_words), len(hyp_words)
        substitutions = deletions = insertions = 0

        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
                i, j = i - 1, j - 1
            elif i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + 1:
                substitutions += 1
                i, j = i - 1, j - 1
            elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
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
                if ref_chars[i - 1] == hyp_chars[j - 1]:
                    dp[i, j] = dp[i - 1, j - 1]
                else:
                    dp[i, j] = min(
                        dp[i - 1, j] + 1,  # deletion
                        dp[i, j - 1] + 1,  # insertion
                        dp[i - 1, j - 1] + 1,  # substitution
                    )

        cer = dp[len(ref_chars), len(hyp_chars)] / max(1, len(ref_chars))
        return cer

    @staticmethod
    def evaluate(
        transcription: str,
        ground_truth: str,
        audio_duration_s: float,
        latency_s: float,
        metadata: dict[str, Any] | None = None,
    ) -> ASRResult:
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
            metadata=metadata or {},
        )

        # Log breakdown
        logger.info("ASR Evaluation:")
        logger.info(f"  WER: {wer:.3f} (S:{sub}, D:{del_}, I:{ins})")
        logger.info(f"  CER: {cer:.3f}")
        logger.info(f"  Latency: {latency_ms:.1f}ms")
        logger.info(f"  RTF: {rtv:.3f}x")

        return result


class ASRBatcher:
    """Batch evaluation for multiple tests."""

    @staticmethod
    def evaluate_batch(results: list[ASRResult]) -> dict[str, Any]:
        """
        Calculate aggregate metrics across multiple ASR results.

        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        if not results:
            return {}

        metrics = {
            "wer_mean": np.mean([r.wer for r in results]),
            "wer_std": np.std([r.wer for r in results]),
            "cer_mean": np.mean([r.cer for r in results]),
            "cer_std": np.std([r.cer for r in results]),
            "latency_ms_mean": np.mean([r.latency_ms for r in results]),
            "latency_ms_std": np.std([r.latency_ms for r in results]),
            "rtv_mean": np.mean([r.rtv for r in results]),
            "rtv_std": np.std([r.rtv for r in results]),
            "num_samples": len(results),
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


# =============================================================================
# Output Quality Diagnosis - Detect failure modes
# =============================================================================

import re
from collections import Counter


def _ngrams(tokens: list[str], n: int = 3) -> list[tuple]:
    """Generate n-grams from token list."""
    if len(tokens) < n:
        return []
    return list(zip(*[tokens[i:] for i in range(n)]))


def repeat_3gram_rate(text: str) -> float:
    """
    Calculate repetition rate based on 3-gram frequency.
    Higher values indicate more repetitive/stuck output.
    """
    tokens = re.findall(r"\w+", text.lower())
    grams = _ngrams(tokens, 3)
    if not grams:
        return 0.0

    counts = Counter(grams)
    repeats = sum(v - 1 for v in counts.values() if v > 1)
    return repeats / max(len(grams), 1)


def diagnose_output_quality(reference: str, hypothesis: str) -> dict[str, Any]:
    """
    Diagnose ASR output quality to detect failure modes.

    Detects:
        - Truncation: model dropped content (length_ratio < 0.7)
        - Hallucination: model inserted content (length_ratio > 1.3)
        - Repetition: model is stuck in loop (unique_ratio < 0.4 or repeat_3gram > 0.2)

    Args:
        reference: Ground truth text
        hypothesis: Model output text

    Returns:
        Dict with diagnostic metrics and failure flags
    """
    ref_words = re.findall(r"\w+", reference)
    hyp_words = re.findall(r"\w+", hypothesis)

    # Core ratios
    length_ratio = len(hyp_words) / max(len(ref_words), 1)
    char_ratio = len(hypothesis) / max(len(reference), 1)

    # Repetition detection
    hyp_words_lower = [w.lower() for w in hyp_words]
    unique_token_ratio = len(set(hyp_words_lower)) / max(len(hyp_words), 1)
    rep3 = repeat_3gram_rate(hypothesis)

    # Failure flags with tightened thresholds
    is_truncated = length_ratio < 0.7
    is_hallucinating = length_ratio > 1.3
    is_repetitive = (unique_token_ratio < 0.4) or (rep3 > 0.2)

    diagnosis = {
        "ref_word_count": len(ref_words),
        "hyp_word_count": len(hyp_words),
        "length_ratio": float(length_ratio),
        "char_ratio": float(char_ratio),
        "unique_token_ratio": float(unique_token_ratio),
        "repeat_3gram_rate": float(rep3),
        "is_truncated": is_truncated,
        "is_hallucinating": is_hallucinating,
        "is_repetitive": is_repetitive,
        "has_failure": is_truncated or is_hallucinating or is_repetitive,
    }

    # Log if failure detected
    if diagnosis["has_failure"]:
        failures = []
        if is_truncated:
            failures.append(f"TRUNCATED (ratio={length_ratio:.2f})")
        if is_hallucinating:
            failures.append(f"HALLUCINATING (ratio={length_ratio:.2f})")
        if is_repetitive:
            failures.append(f"REPETITIVE (unique={unique_token_ratio:.2f}, 3gram={rep3:.2f})")
        logger.warning(f"Output quality issues: {', '.join(failures)}")

    return diagnosis


def diagnose_output_no_reference(
    hypothesis: str, expected_chars_per_sec: float = 15.0, audio_duration_s: float = 0.0
) -> dict[str, Any]:
    """
    Diagnose output quality when no ground truth is available.
    Uses heuristics based on expected output density.

    Args:
        hypothesis: Model output text
        expected_chars_per_sec: Expected character rate (default ~15 for English speech)
        audio_duration_s: Audio duration in seconds

    Returns:
        Dict with diagnostic metrics
    """
    hyp_words = re.findall(r"\w+", hypothesis)
    hyp_words_lower = [w.lower() for w in hyp_words]

    unique_token_ratio = len(set(hyp_words_lower)) / max(len(hyp_words), 1)
    rep3 = repeat_3gram_rate(hypothesis)

    # Coverage proxy: chars per second of audio
    if audio_duration_s > 0:
        chars_per_sec = len(hypothesis) / audio_duration_s
        coverage_ratio = chars_per_sec / expected_chars_per_sec
    else:
        chars_per_sec = 0.0
        coverage_ratio = 0.0

    is_repetitive = (unique_token_ratio < 0.4) or (rep3 > 0.2)
    is_sparse = coverage_ratio < 0.5 if audio_duration_s > 0 else False

    return {
        "hyp_word_count": len(hyp_words),
        "hyp_char_count": len(hypothesis),
        "unique_token_ratio": float(unique_token_ratio),
        "repeat_3gram_rate": float(rep3),
        "chars_per_sec": float(chars_per_sec),
        "coverage_ratio": float(coverage_ratio),
        "is_repetitive": is_repetitive,
        "is_sparse": is_sparse,
        "has_failure": is_repetitive or is_sparse,
    }
