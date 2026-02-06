"""
TTS evaluation metrics for model comparison.
Implements audio similarity, naturalness, and timing metrics.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    """Container for TTS evaluation results."""

    audio_path: str
    text: str
    audio_duration_s: float
    latency_ms: float
    similarity_score: float  # Audio similarity with reference
    metadata: dict[str, Any]


class TTSMetrics:
    """Calculate TTS evaluation metrics."""

    @staticmethod
    def calculate_mfcc_similarity(audio1: np.ndarray, audio2: np.ndarray, sr: int) -> float:
        """
        Calculate MFCC-based audio similarity.
        Higher values indicate more similar audio.
        """
        import torch
        import torchaudio

        # Convert to tensors
        audio1_tensor: torch.Tensor
        audio2_tensor: torch.Tensor
        if isinstance(audio1, np.ndarray):
            audio1_tensor = torch.from_numpy(audio1).float()
        else:
            audio1_tensor = audio1  # type: ignore[assignment]
        if isinstance(audio2, np.ndarray):
            audio2_tensor = torch.from_numpy(audio2).float()
        else:
            audio2_tensor = audio2  # type: ignore[assignment]

        # Ensure same length
        min_len = min(len(audio1_tensor), len(audio2_tensor))
        audio1_tensor = audio1_tensor[:min_len]
        audio2_tensor = audio2_tensor[:min_len]

        # Calculate MFCCs
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sr, n_mfcc=13, melkwargs={"n_mels": 40, "n_fft": 512}
        )

        mfcc1 = mfcc_transform(audio1_tensor.unsqueeze(0))
        mfcc2 = mfcc_transform(audio2_tensor.unsqueeze(0))

        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(mfcc1.flatten(), mfcc2.flatten(), dim=0)

        return similarity.item()

    @staticmethod
    def calculate_timing_metrics(
        generated_audio: np.ndarray, reference_audio: np.ndarray, sr: int
    ) -> dict[str, float]:
        """
        Calculate timing-related metrics.
        """
        gen_duration = len(generated_audio) / sr
        ref_duration = len(reference_audio) / sr

        duration_ratio = gen_duration / max(0.001, ref_duration)
        duration_diff = gen_duration - ref_duration

        return {
            "duration_ratio": duration_ratio,
            "duration_diff_s": duration_diff,
            "generated_duration_s": gen_duration,
            "reference_duration_s": ref_duration,
        }

    @staticmethod
    def evaluate(
        generated_audio: np.ndarray,
        reference_audio: np.ndarray,
        text: str,
        sr: int,
        latency_s: float,
        metadata: dict[str, Any] | None = None,
    ) -> TTSResult:
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
        similarity = TTSMetrics.calculate_mfcc_similarity(generated_audio, reference_audio, sr)

        # Calculate timing metrics
        timing = TTSMetrics.calculate_timing_metrics(generated_audio, reference_audio, sr)

        latency_ms = latency_s * 1000

        result = TTSResult(
            audio_path="generated_audio",  # Could save to file
            text=text,
            audio_duration_s=timing["generated_duration_s"],
            latency_ms=latency_ms,
            similarity_score=similarity,
            metadata={**timing, **(metadata or {})},
        )

        logger.info("TTS Evaluation:")
        logger.info(f"  Similarity: {similarity:.3f}")
        logger.info(f"  Duration ratio: {timing['duration_ratio']:.3f}")
        logger.info(f"  Latency: {latency_ms:.1f}ms")

        return result


class VoiceQualityMetrics:
    """Assess voice quality characteristics."""

    @staticmethod
    def calculate_spectral_centroid(audio: np.ndarray, sr: int) -> float:
        """Calculate spectral centroid (brightness)."""
        import torch

        audio_tensor: torch.Tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio  # type: ignore[assignment]

        # Simple approximation using FFT
        fft = torch.fft.fft(audio_tensor.unsqueeze(0))
        magnitude = torch.abs(fft).mean(dim=0)

        # Centroid calculation
        freqs = torch.fft.fftfreq(len(audio_tensor), 1 / sr)[: len(audio_tensor) // 2]
        magnitude_half = magnitude[: len(audio_tensor) // 2]

        if magnitude_half.sum() > 0:
            centroid = (freqs * magnitude_half).sum() / magnitude_half.sum()
        else:
            centroid = 0.0

        return centroid.item()

    @staticmethod
    def calculate_energy(audio: np.ndarray) -> float:
        """Calculate RMS energy."""
        import torch

        audio_tensor: torch.Tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio  # type: ignore[assignment]
        return torch.sqrt(torch.mean(audio_tensor**2)).item()

    @staticmethod
    def assess_voice_characteristics(audio: np.ndarray, sr: int) -> dict[str, float]:
        """Assess various voice quality metrics."""
        return {
            "spectral_centroid": VoiceQualityMetrics.calculate_spectral_centroid(audio, sr),
            "rms_energy": VoiceQualityMetrics.calculate_energy(audio),
            "duration_s": len(audio) / sr,
        }


# =============================================================================
# TTS Failure Detection - Production gates
# =============================================================================


def detect_audio_issues(audio: np.ndarray, sr: int) -> dict[str, Any]:
    """
    Detect TTS failure modes for production gating.

    Detects:
        - Clipping: audio exceeds safe amplitude
        - Silence: too much of the audio is silent
        - DC offset: broken pipeline producing offset audio

    Args:
        audio: Audio waveform (numpy array)
        sr: Sample rate

    Returns:
        Dict with failure metrics and flags
    """
    # Ensure float32 and proper scaling
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0

    audio = np.asarray(audio, dtype=np.float32)

    # Basic amplitude metrics
    peak_amplitude = float(np.abs(audio).max()) if len(audio) > 0 else 0.0
    rms_loudness = float(np.sqrt(np.mean(audio**2))) if len(audio) > 0 else 0.0
    dc_offset = float(np.mean(audio)) if len(audio) > 0 else 0.0

    # Clipping detection (vectorized)
    clipping_ratio = float(np.mean(np.abs(audio) > 0.99)) if len(audio) > 0 else 0.0

    # Silence detection (vectorized with frame stride)
    # Use 25ms frames with 10ms hop
    frame_size = int(0.025 * sr)
    hop_size = int(0.010 * sr)

    if len(audio) >= frame_size:
        # Vectorized frame energy calculation using stride tricks
        n_frames = (len(audio) - frame_size) // hop_size + 1

        # Compute frame energies efficiently
        frame_energies = np.array(
            [
                np.sqrt(np.mean(audio[i * hop_size : i * hop_size + frame_size] ** 2))
                for i in range(min(n_frames, 1000))  # Cap at 1000 frames for speed
            ]
        )

        silence_threshold = 0.01
        silence_ratio = float(np.mean(frame_energies < silence_threshold))
    else:
        silence_ratio = 1.0 if rms_loudness < 0.01 else 0.0

    # Duration
    duration_s = len(audio) / sr if sr > 0 else 0.0

    # Failure flags
    is_mostly_silent = silence_ratio > 0.5
    has_clipping = clipping_ratio > 0.01
    has_dc_offset = abs(dc_offset) > 0.1
    is_empty = len(audio) == 0 or duration_s < 0.1

    issues = {
        "peak_amplitude": peak_amplitude,
        "rms_loudness": rms_loudness,
        "dc_offset": dc_offset,
        "clipping_ratio": clipping_ratio,
        "silence_ratio": silence_ratio,
        "duration_s": duration_s,
        "is_mostly_silent": is_mostly_silent,
        "has_clipping": has_clipping,
        "has_dc_offset": has_dc_offset,
        "is_empty": is_empty,
        "has_failure": is_mostly_silent or has_clipping or has_dc_offset or is_empty,
    }

    # Log if failure detected
    if issues["has_failure"]:
        failures = []
        if is_empty:
            failures.append("EMPTY")
        if is_mostly_silent:
            failures.append(f"SILENT ({silence_ratio:.1%})")
        if has_clipping:
            failures.append(f"CLIPPING ({clipping_ratio:.1%})")
        if has_dc_offset:
            failures.append(f"DC_OFFSET ({dc_offset:.3f})")
        logger.warning(f"TTS audio issues: {', '.join(failures)}")

    return issues


def format_tts_result(result: TTSResult) -> str:
    """Format TTS result for logging."""
    return (
        f"Similarity: {result.similarity_score:.3f}, "
        f"Duration: {result.audio_duration_s:.1f}s, "
        f"Latency: {result.latency_ms:.1f}ms"
    )
