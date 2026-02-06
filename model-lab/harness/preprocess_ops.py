"""
Preprocessing Operators - Audited audio transformation chain.

Each operator:
- Takes audio as input, returns transformed audio
- Records: name, version, params, in_audio_hash, out_audio_hash, metrics
- Is pure audio-to-audio (no side effects)

Operators:
- trim_silence: Remove silence from start/end using VAD or energy
- normalize_loudness: LUFS normalization using pyloudnorm
- resample: Force canonical sample rate (usually no-op after ingest)

Usage:
    from harness.preprocess_ops import run_preprocessing_chain, TrimSilence, NormalizeLoudness

    results = run_preprocessing_chain(
        audio, sample_rate,
        operators=["trim_silence", "normalize_loudness"]
    )
    # results[-1].audio is the final audio
    # Each result has in_audio_hash, out_audio_hash, metrics
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

OPERATOR_VERSION = "1.0.0"


@dataclass
class OperatorResult:
    """Result of a single preprocessing operator."""

    name: str  # Operator name
    version: str  # Operator version
    params: dict[str, Any]  # Parameters used
    in_audio_hash: str  # Hash of input audio
    out_audio_hash: str  # Hash of output audio
    metrics: dict[str, float]  # Operator-specific metrics
    audio: np.ndarray  # Transformed audio
    sample_rate: int  # Sample rate (may change for resample)
    duration_in_s: float  # Input duration
    duration_out_s: float  # Output duration

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for artifact storage (excludes audio)."""
        return {
            "name": self.name,
            "version": self.version,
            "params": self.params,
            "in_audio_hash": self.in_audio_hash,
            "out_audio_hash": self.out_audio_hash,
            "metrics": self.metrics,
            "duration_in_s": self.duration_in_s,
            "duration_out_s": self.duration_out_s,
        }


def compute_audio_hash(audio: np.ndarray, length: int = 16) -> str:
    """Compute SHA256 hash of audio PCM bytes."""
    audio_bytes = audio.astype(np.float32).tobytes()
    return hashlib.sha256(audio_bytes).hexdigest()[:length]


class Operator(ABC):
    """Base class for preprocessing operators."""

    name: str = "base"
    version: str = OPERATOR_VERSION

    @abstractmethod
    def process(
        self, audio: np.ndarray, sr: int, **kwargs
    ) -> tuple[np.ndarray, int, dict[str, float]]:
        """
        Process audio and return (transformed_audio, sample_rate, metrics).

        Args:
            audio: Input audio (mono, float32)
            sr: Sample rate
            **kwargs: Operator-specific parameters

        Returns:
            Tuple of (output_audio, output_sr, metrics_dict)
        """
        pass

    def run(self, audio: np.ndarray, sr: int, **kwargs) -> OperatorResult:
        """Run operator and return full result with hashes."""
        in_hash = compute_audio_hash(audio)
        duration_in = len(audio) / sr

        out_audio, out_sr, metrics = self.process(audio, sr, **kwargs)

        out_hash = compute_audio_hash(out_audio)
        duration_out = len(out_audio) / out_sr

        return OperatorResult(
            name=self.name,
            version=self.version,
            params=kwargs,
            in_audio_hash=in_hash,
            out_audio_hash=out_hash,
            metrics=metrics,
            audio=out_audio,
            sample_rate=out_sr,
            duration_in_s=duration_in,
            duration_out_s=duration_out,
        )


class TrimSilence(Operator):
    """
    Remove silence from start and end of audio.

    Uses energy-based detection with configurable threshold.
    For more accurate results, can use VAD when available.
    """

    name = "trim_silence"

    def process(
        self,
        audio: np.ndarray,
        sr: int,
        threshold_db: float = -40.0,
        min_silence_ms: int = 100,
        **kwargs,
    ) -> tuple[np.ndarray, int, dict[str, float]]:
        """
        Trim silence from audio edges.

        Args:
            threshold_db: Silence threshold in dB (default -40)
            min_silence_ms: Minimum silence duration to consider
        """
        # Convert to dB scale
        eps = 1e-10
        audio_db = 20 * np.log10(np.abs(audio) + eps)

        # Find non-silent regions
        is_sound = audio_db > threshold_db

        if not np.any(is_sound):
            # All silence, return minimal audio
            metrics = {
                "trimmed_start_ms": 0,
                "trimmed_end_ms": len(audio) / sr * 1000,
                "total_trimmed_ms": len(audio) / sr * 1000,
            }
            return audio[: int(sr * 0.1)], sr, metrics  # Keep 100ms

        # Apply minimum duration filter using morphological operations
        min_samples = int(min_silence_ms * sr / 1000)

        # Simple approach: find first and last non-silent sample
        nonzero_indices = np.where(is_sound)[0]
        start_idx = max(0, nonzero_indices[0] - min_samples)
        end_idx = min(len(audio), nonzero_indices[-1] + min_samples)

        trimmed_audio = audio[start_idx:end_idx]

        trimmed_start_ms = start_idx / sr * 1000
        trimmed_end_ms = (len(audio) - end_idx) / sr * 1000

        metrics = {
            "trimmed_start_ms": round(trimmed_start_ms, 1),
            "trimmed_end_ms": round(trimmed_end_ms, 1),
            "total_trimmed_ms": round(trimmed_start_ms + trimmed_end_ms, 1),
            "threshold_db": threshold_db,
        }

        logger.info(f"trim_silence: removed {metrics['total_trimmed_ms']:.0f}ms")

        return trimmed_audio.astype(np.float32), sr, metrics


class NormalizeLoudness(Operator):
    """
    Normalize audio loudness to target LUFS using pyloudnorm.

    BS.1770-4 integrated loudness normalization.
    """

    name = "normalize_loudness"

    def process(
        self,
        audio: np.ndarray,
        sr: int,
        target_lufs: float = -23.0,
        peak_limit_db: float = -1.0,
        **kwargs,
    ) -> tuple[np.ndarray, int, dict[str, float]]:
        """
        Normalize loudness to target LUFS.

        Args:
            target_lufs: Target integrated loudness (default -23 LUFS, EBU R128)
            peak_limit_db: Peak limiting threshold (default -1 dBFS)
        """
        try:
            import pyloudnorm as pyln
        except ImportError:
            logger.warning("pyloudnorm not installed, skipping normalization")
            return audio, sr, {"error": "pyloudnorm not installed"}

        # Measure input loudness
        meter = pyln.Meter(sr)

        try:
            input_lufs = meter.integrated_loudness(audio)
        except Exception as e:
            logger.warning(f"Could not measure loudness: {e}")
            return audio, sr, {"error": str(e)}

        # Handle silence or very quiet audio
        if np.isinf(input_lufs) or input_lufs < -70:
            metrics = {
                "input_lufs": float("-inf"),
                "output_lufs": float("-inf"),
                "gain_db": 0,
                "peak_limited": False,
                "target_lufs": target_lufs,
            }
            return audio, sr, metrics

        # Normalize to target
        try:
            normalized = pyln.normalize.loudness(audio, input_lufs, target_lufs)
        except Exception as e:
            logger.warning(f"Normalization failed: {e}")
            return audio, sr, {"error": str(e)}

        # Peak limiting
        peak_linear = 10 ** (peak_limit_db / 20)
        peak_before = np.max(np.abs(normalized))
        peak_limited = False

        if peak_before > peak_linear:
            normalized = normalized * (peak_linear / peak_before)
            peak_limited = True

        # Measure output
        try:
            output_lufs = meter.integrated_loudness(normalized)
        except:
            output_lufs = target_lufs  # Assume success

        gain_db = target_lufs - input_lufs if not np.isinf(input_lufs) else 0

        metrics = {
            "input_lufs": round(input_lufs, 1),
            "output_lufs": round(output_lufs, 1) if not np.isinf(output_lufs) else target_lufs,
            "gain_db": round(gain_db, 1),
            "peak_limited": peak_limited,
            "target_lufs": target_lufs,
            "peak_before": round(20 * np.log10(peak_before + 1e-10), 1),
        }

        logger.info(f"normalize_loudness: {input_lufs:.1f} -> {output_lufs:.1f} LUFS")

        return normalized.astype(np.float32), sr, metrics


class Resample(Operator):
    """
    Resample audio to target sample rate.

    Usually a no-op after ingest (which already canonicalizes to 16kHz).
    Kept for explicit pipeline control.
    """

    name = "resample"

    def process(
        self, audio: np.ndarray, sr: int, target_sr: int = 16000, **kwargs
    ) -> tuple[np.ndarray, int, dict[str, float]]:
        """
        Resample to target sample rate.

        Args:
            target_sr: Target sample rate (default 16000)
        """
        if sr == target_sr:
            metrics = {
                "input_sr": sr,
                "output_sr": target_sr,
                "resampled": False,
            }
            return audio, sr, metrics

        # Resample using scipy or librosa
        try:
            import librosa

            resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            from scipy import signal

            num_samples = int(len(audio) * target_sr / sr)
            resampled = signal.resample(audio, num_samples)

        metrics = {
            "input_sr": sr,
            "output_sr": target_sr,
            "resampled": True,
            "ratio": round(target_sr / sr, 3),
        }

        logger.info(f"resample: {sr} -> {target_sr} Hz")

        return resampled.astype(np.float32), target_sr, metrics


class NormalizeVolume(Operator):
    """Peak normalization to target dBFS."""

    name = "normalize_volume"

    def process(self, audio: np.ndarray, sr: int, target_dbfs: float = -1.0, **kwargs):
        peak = float(np.max(np.abs(audio))) + 1e-12
        target_linear = 10 ** (target_dbfs / 20)
        gain = target_linear / peak
        out = (audio * gain).astype(np.float32)
        metrics = {
            "peak_before_db": round(20 * np.log10(peak), 1),
            "peak_after_db": round(20 * np.log10(float(np.max(np.abs(out))) + 1e-12), 1),
            "gain_db": round(20 * np.log10(gain), 1),
            "target_dbfs": target_dbfs,
        }
        return out, sr, metrics


class ExtractChannel(Operator):
    """Extract single channel from multichannel audio."""

    name = "extract_channel"

    def process(self, audio: np.ndarray, sr: int, channel: int = 0, **kwargs):
        if audio.ndim == 1:
            return audio, sr, {"no_op": True, "reason": "already_mono"}
        c = audio.shape[1] if audio.ndim == 2 else 1
        if channel < 0 or channel >= c:
            raise ValueError(f"channel out of range: {channel}, available=0..{c - 1}")
        out = audio[:, channel].astype(np.float32)
        return out, sr, {"channels_in": c, "channel_selected": channel}


class Denoise(Operator):
    """Background noise reduction using noisereduce (optional dependency)."""

    name = "denoise"

    def process(self, audio: np.ndarray, sr: int, prop_decrease: float = 1.0, **kwargs):
        try:
            import noisereduce as nr
        except ImportError:
            logger.warning("noisereduce not installed, skipping denoise")
            return audio, sr, {"skipped": True, "reason": "noisereduce_not_installed"}

        out = nr.reduce_noise(y=audio, sr=sr, prop_decrease=prop_decrease)
        return out.astype(np.float32), sr, {"skipped": False, "prop_decrease": prop_decrease}


class SpeedAdjust(Operator):
    """Adjust playback speed without pitch change (optional librosa dependency)."""

    name = "speed"

    def process(self, audio: np.ndarray, sr: int, factor: float = 1.0, **kwargs):
        if abs(factor - 1.0) < 0.01:
            return audio, sr, {"no_op": True, "factor": factor}
        try:
            import librosa

            out = librosa.effects.time_stretch(audio, rate=factor)
            new_duration = len(out) / sr
            orig_duration = len(audio) / sr
            return (
                out.astype(np.float32),
                sr,
                {
                    "factor": factor,
                    "duration_before_s": round(orig_duration, 2),
                    "duration_after_s": round(new_duration, 2),
                },
            )
        except ImportError:
            logger.warning("librosa not installed, skipping speed adjustment")
            return audio, sr, {"skipped": True, "reason": "librosa_not_installed"}


# Operator registry
OPERATORS: dict[str, Operator] = {
    "trim_silence": TrimSilence(),
    "normalize_loudness": NormalizeLoudness(),
    "normalize_volume": NormalizeVolume(),
    "resample": Resample(),
    "extract_channel": ExtractChannel(),
    "denoise": Denoise(),
    "speed": SpeedAdjust(),
}


def parse_operator_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """
    Parse operator specification string.

    Format: "name" or "name:param=value,param2=value2"

    Examples:
        "trim_silence"
        "normalize_loudness:target_lufs=-16"
        "resample:target_sr=22050"
    """
    if ":" not in spec:
        return spec, {}

    name, params_str = spec.split(":", 1)
    params = {}

    for param in params_str.split(","):
        if "=" in param:
            key, value = param.split("=", 1)
            # Try to parse as number
            try:
                if "." in value:
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except ValueError:
                params[key] = value

    return name, params


def run_preprocessing_chain(
    audio: np.ndarray,
    sample_rate: int,
    operators: list[str],
) -> list[OperatorResult]:
    """
    Run a chain of preprocessing operators.

    Args:
        audio: Input audio (mono, float32)
        sample_rate: Sample rate
        operators: List of operator specs (e.g., ["trim_silence", "normalize_loudness"])

    Returns:
        List of OperatorResult, one per operator.
        Final audio is results[-1].audio
    """
    if not operators:
        return []

    results = []
    current_audio = audio
    current_sr = sample_rate

    for spec in operators:
        name, params = parse_operator_spec(spec)

        if name not in OPERATORS:
            raise ValueError(f"Unknown operator: {name}. Available: {list(OPERATORS.keys())}")

        op = OPERATORS[name]
        result = op.run(current_audio, current_sr, **params)
        results.append(result)

        current_audio = result.audio
        current_sr = result.sample_rate

    return results


def results_to_artifact_section(results: list[OperatorResult]) -> list[dict[str, Any]]:
    """Convert operator results to artifact-friendly format."""
    return [r.to_dict() for r in results]
