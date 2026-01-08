"""
TTS evaluation metrics for model comparison.
Implements audio similarity, naturalness, and timing metrics.
"""

import numpy as np
import torch
import torchaudio
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    """Container for TTS evaluation results."""
    audio_path: str
    text: str
    audio_duration_s: float
    latency_ms: float
    similarity_score: float  # Audio similarity with reference
    metadata: Dict[str, Any]


class TTSMetrics:
    """Calculate TTS evaluation metrics."""

    @staticmethod
    def calculate_mfcc_similarity(audio1: np.ndarray,
                                  audio2: np.ndarray,
                                  sr: int) -> float:
        """
        Calculate MFCC-based audio similarity.
        Higher values indicate more similar audio.
        """
        # Convert to tensors
        if isinstance(audio1, np.ndarray):
            audio1 = torch.from_numpy(audio1).float()
        if isinstance(audio2, np.ndarray):
            audio2 = torch.from_numpy(audio2).float()

        # Ensure same length
        min_len = min(len(audio1), len(audio2))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]

        # Calculate MFCCs
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=13,
            melkwargs={'n_mels': 40, 'n_fft': 512}
        )

        mfcc1 = mfcc_transform(audio1.unsqueeze(0))
        mfcc2 = mfcc_transform(audio2.unsqueeze(0))

        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            mfcc1.flatten(),
            mfcc2.flatten(),
            dim=0
        )

        return similarity.item()

    @staticmethod
    def calculate_timing_metrics(generated_audio: np.ndarray,
                                 reference_audio: np.ndarray,
                                 sr: int) -> Dict[str, float]:
        """
        Calculate timing-related metrics.
        """
        gen_duration = len(generated_audio) / sr
        ref_duration = len(reference_audio) / sr

        duration_ratio = gen_duration / max(0.001, ref_duration)
        duration_diff = gen_duration - ref_duration

        return {
            'duration_ratio': duration_ratio,
            'duration_diff_s': duration_diff,
            'generated_duration_s': gen_duration,
            'reference_duration_s': ref_duration
        }

    @staticmethod
    def evaluate(generated_audio: np.ndarray,
                 reference_audio: np.ndarray,
                 text: str,
                 sr: int,
                 latency_s: float,
                 metadata: Dict[str, Any] = None) -> TTSResult:
        """
        Comprehensive TTS evaluation.

        Args:
            generated_audio: Synthesized audio
            reference_audio: Reference recording
            text: Input text
            sr: Sample rate
            latency_s: Processing time
            metadata: Additional metadata

        Returns:
            TTSResult with all metrics
        """
        # Calculate similarity
        similarity = TTSMetrics.calculate_mfcc_similarity(
            generated_audio, reference_audio, sr
        )

        # Calculate timing metrics
        timing = TTSMetrics.calculate_timing_metrics(
            generated_audio, reference_audio, sr
        )

        latency_ms = latency_s * 1000

        result = TTSResult(
            audio_path="generated_audio",  # Could save to file
            text=text,
            audio_duration_s=timing['generated_duration_s'],
            latency_ms=latency_ms,
            similarity_score=similarity,
            metadata={**timing, **(metadata or {})}
        )

        logger.info(f"TTS Evaluation:")
        logger.info(f"  Similarity: {similarity:.3f}")
        logger.info(f"  Duration ratio: {timing['duration_ratio']:.3f}")
        logger.info(f"  Latency: {latency_ms:.1f}ms")

        return result


class VoiceQualityMetrics:
    """Assess voice quality characteristics."""

    @staticmethod
    def calculate_spectral_centroid(audio: np.ndarray, sr: int) -> float:
        """Calculate spectral centroid (brightness)."""
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        # Simple approximation using FFT
        fft = torch.fft.fft(audio.unsqueeze(0))
        magnitude = torch.abs(fft).mean(dim=0)

        # Centroid calculation
        freqs = torch.fft.fftfreq(len(audio), 1/sr)[:len(audio)//2]
        magnitude_half = magnitude[:len(audio)//2]

        if magnitude_half.sum() > 0:
            centroid = (freqs * magnitude_half).sum() / magnitude_half.sum()
        else:
            centroid = 0.0

        return centroid.item()

    @staticmethod
    def calculate_energy(audio: np.ndarray) -> float:
        """Calculate RMS energy."""
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        return torch.sqrt(torch.mean(audio**2)).item()

    @staticmethod
    def assess_voice_characteristics(audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Assess various voice quality metrics."""
        return {
            'spectral_centroid': VoiceQualityMetrics.calculate_spectral_centroid(audio, sr),
            'rms_energy': VoiceQualityMetrics.calculate_energy(audio),
            'duration_s': len(audio) / sr
        }


def format_tts_result(result: TTSResult) -> str:
    """Format TTS result for logging."""
    return (
        f"Similarity: {result.similarity_score:.3f}, "
        f"Duration: {result.audio_duration_s:.1f}s, "
        f"Latency: {result.latency_ms:.1f}ms"
    )