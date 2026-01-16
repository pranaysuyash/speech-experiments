"""
Shared audio I/O module for all model testing.
Ensures consistent audio loading, preprocessing, and format handling across models.
"""

import hashlib
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_input_hash(file_path: Path, length: int = 2 * 1024 * 1024) -> str:
    """Compute hash of the first 2MB of a file (for identifying inputs)."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        sha256.update(f.read(length))
    return sha256.hexdigest()


class AudioLoader:
    """Standardized audio loading for model testing."""

    # Canonical sample rates for different models
    SAMPLE_RATES = {
        'lfm2_5_audio': 24000,
        'whisper': 16000,
        'seamlessm4t': 16000,
    }

    def __init__(self, target_sample_rate: Optional[int] = None):
        """
        Initialize audio loader.

        Args:
            target_sample_rate: Target sample rate (None = use model default)
        """
        self.target_sample_rate = target_sample_rate

    def load_audio(self,
                   audio_path: Path,
                   model_type: str = 'lfm2_5_audio',
                   convert_to_mono: bool = True) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """
        Load audio file with standardized preprocessing.

        Args:
            audio_path: Path to audio file
            model_type: Model type for sample rate selection
            convert_to_mono: Convert stereo to mono

        Returns:
            audio: Audio array (samples, channels)
            sample_rate: Sample rate in Hz
            metadata: Audio metadata
        """
        import soundfile as sf  # Lazy import

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio
        try:
            audio, sample_rate = sf.read(audio_path)
        except Exception as e:
            raise IOError(f"Failed to load audio file {audio_path}: {e}")

        # Handle channel conversion
        if audio.ndim > 1:
            if convert_to_mono:
                audio = audio.mean(axis=1)
                logger.info(f"Converted to mono: {audio_path.name}")
            else:
                audio = audio.T  # (channels, samples)

        # Capture original sample rate BEFORE resampling
        original_sample_rate = sample_rate

        # Resample if needed
        target_sr = self.target_sample_rate or self.SAMPLE_RATES.get(model_type, 16000)
        if sample_rate != target_sr:
            audio = self._resample_audio(audio, sample_rate, target_sr)
            sample_rate = target_sr
            logger.info(f"Resampled to {target_sr}Hz: {audio_path.name}")

        # Metadata - use original_sample_rate, not post-resample value
        metadata = {
            'original_sample_rate': original_sample_rate,
            'sample_rate': sample_rate,
            'duration_seconds': len(audio) / sample_rate,
            'num_samples': len(audio),
            'channels': 1 if audio.ndim == 1 else audio.shape[0],
            'path': str(audio_path)
        }

        logger.info(f"Loaded {audio_path.name}: {metadata['duration_seconds']:.1f}s @ {sample_rate}Hz")
        return audio, sample_rate, metadata

    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio using torchaudio."""
        import torch
        import torchaudio

        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
            if audio.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # (1, samples)
        else:
            audio_tensor = audio

        # Resample
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        resampled = resampler(audio_tensor)

        # Convert back to numpy
        return resampled.squeeze().numpy()

    def save_audio(self, audio: np.ndarray, sample_rate: int, output_path: Path):
        """Save audio to file."""
        import soundfile as sf # Lazy import
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, sample_rate)
        logger.info(f"Saved audio to {output_path}")


class GroundTruthLoader:
    """Load ground truth text for evaluation."""

    @staticmethod
    def load_text(text_path: Path) -> str:
        """Load ground truth text from file."""
        if not text_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {text_path}")

        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        logger.info(f"Loaded ground truth: {len(text)} chars from {text_path.name}")
        return text

    @staticmethod
    def load_pair(audio_path: Path, text_path: Path) -> Tuple[np.ndarray, int, str]:
        """Load audio and corresponding ground truth text."""
        loader = AudioLoader()
        audio, sr, _ = loader.load_audio(audio_path)
        text = GroundTruthLoader.load_text(text_path)
        return audio, sr, text


def create_canonical_test_set():
    """
    Create canonical test audio files if they don't exist.
    This is a placeholder - actual implementation would generate test audio.
    """
    logger.warning("Canonical test set creation not implemented yet")
    logger.info("Please manually create test audio files in data/audio/canonical/")