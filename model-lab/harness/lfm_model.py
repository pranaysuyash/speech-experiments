"""
Complete LFM-2.5-Audio model manager with all capabilities.
Handles local model loading, inference, and advanced evaluation.
"""

import gc
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import torchaudio

# Import Liquid Audio library
try:
    from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor, LFMModality

    LIQUID_AUDIO_AVAILABLE = True
except ImportError:
    LIQUID_AUDIO_AVAILABLE = False
    warnings.warn(
        "Liquid Audio library not available. Install with: pip install liquid-audio", stacklevel=2
    )


class LFMModelManager:
    """Complete manager for LFM-2.5-Audio model with all capabilities."""

    def __init__(self, repo_id: str = "LiquidAI/LFM2.5-Audio-1.5B", device: str = None):
        self.repo_id = repo_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.is_loaded = False
        self.loading_metrics = {}

    def load_model(self, precision: str = "float32") -> dict[str, Any]:
        """Load model with comprehensive metrics and error handling."""

        if not LIQUID_AUDIO_AVAILABLE:
            raise RuntimeError(
                "Liquid Audio library not available. Install with: pip install liquid-audio"
            )

        print(f"Loading LFM-2.5-Audio model from {self.repo_id}")
        print(f"Device: {self.device}")
        print(f"Target precision: {precision}")

        # Get baseline system state
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1e6

        start_time = time.time()

        try:
            # Set precision
            if precision == "float16" and torch.cuda.is_available():
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

            print(f"Using torch dtype: {torch_dtype}")

            # Load processor
            print("Loading processor...")
            processor_start = time.time()
            self.processor = LFM2AudioProcessor.from_pretrained(
                self.repo_id, torch_dtype=torch_dtype
            )
            self.processor.eval()
            processor_time = time.time() - processor_start

            # Load model
            print("Loading model...")
            model_start = time.time()
            self.model = LFM2AudioModel.from_pretrained(self.repo_id, torch_dtype=torch_dtype)
            self.model.eval()

            # Move to device
            if self.device != "cpu":
                print(f"Moving model to {self.device}...")
                self.model = self.model.to(self.device)

            model_time = time.time() - model_start
            total_time = time.time() - start_time

            # Get post-loading metrics
            memory_after = process.memory_info().rss / 1e6

            self.loading_metrics = {
                "processor_load_time": processor_time,
                "model_load_time": model_time,
                "total_load_time": total_time,
                "memory_used_mb": memory_after - memory_before,
                "memory_after_mb": memory_after,
                "precision": str(torch_dtype).split(".")[-1],
                "device": self.device,
            }

            self.is_loaded = True

            print("✓ Model loaded successfully!")
            print(f"  Total load time: {total_time:.1f}s")
            print(f"  Memory used: {self.loading_metrics['memory_used_mb']:.1f}MB")
            print(f"  Precision: {self.loading_metrics['precision']}")

            return {
                "model": self.model,
                "processor": self.processor,
                "device": self.device,
                "loading_metrics": self.loading_metrics,
                "repo_id": self.repo_id,
            }

        except Exception as e:
            total_time = time.time() - start_time
            print(f"✗ Model loading failed: {e}")
            print(f"Failed after: {total_time:.1f}s")

            print("\nTROUBLESHOOTING:")
            print("1. Check HuggingFace Hub connectivity")
            print("2. Verify sufficient GPU memory")
            print("3. Check CUDA availability")
            print("4. Ensure liquid-audio library is installed")

            raise RuntimeError(f"Model loading failed: {e}") from e

    def unload_model(self):
        """Unload model and free resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
        self.is_loaded = False
        print("✓ Model unloaded and resources freed")

    def get_model_info(self) -> dict[str, Any]:
        """Get comprehensive model information."""
        if not self.is_loaded:
            return {"error": "Model not loaded"}

        info = {
            "repo_id": self.repo_id,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "loading_metrics": self.loading_metrics,
            "torch_version": torch.__version__,
            "torchaudio_version": torchaudio.__version__,
        }

        # Add model-specific info if available
        if hasattr(self.model, "config"):
            config = self.model.config
            info.update(
                {
                    "model_type": getattr(config, "model_type", "unknown"),
                    "hidden_size": getattr(config, "hidden_size", None),
                    "num_attention_heads": getattr(config, "num_attention_heads", None),
                    "num_hidden_layers": getattr(config, "num_hidden_layers", None),
                    "vocab_size": getattr(config, "vocab_size", None),
                }
            )

        return info


class AdvancedAudioProcessor:
    """Advanced audio processing for LFM model input with quality metrics."""

    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
        self.resamplers = {}

    def process_audio(
        self, audio_path: str, normalize: bool = True
    ) -> tuple[torch.Tensor, int, dict[str, Any]]:
        """Process audio with comprehensive quality analysis."""

        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio {audio_path}: {e}") from e

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            if sample_rate not in self.resamplers:
                self.resamplers[sample_rate] = torchaudio.transforms.Resample(
                    sample_rate, self.target_sample_rate
                )
            waveform = self.resamplers[sample_rate](waveform)
            sample_rate = self.target_sample_rate

        # Advanced normalization
        if normalize:
            waveform = self.advanced_normalize(waveform)

        # Calculate quality metrics
        quality_metrics = self.calculate_quality_metrics(waveform, sample_rate)

        return waveform, sample_rate, quality_metrics

    def advanced_normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        """Advanced normalization with multiple strategies."""
        # Peak normalization
        peak_norm = waveform / waveform.abs().max()

        # RMS normalization
        rms = torch.sqrt(torch.mean(peak_norm**2))
        if rms > 0:
            rms_norm = peak_norm / (rms * 3.0)
            # Clip to prevent overflow
            rms_norm = torch.clamp(rms_norm, -1.0, 1.0)
            return rms_norm

        return peak_norm

    def calculate_quality_metrics(self, waveform: torch.Tensor, sample_rate: int) -> dict[str, Any]:
        """Calculate comprehensive audio quality metrics."""
        # Basic metrics
        duration = waveform.shape[1] / sample_rate
        rms_level = torch.sqrt(torch.mean(waveform**2)).item()
        peak_level = waveform.abs().max().item()
        dynamic_range = 20 * np.log10(peak_level / (rms_level + 1e-10)) if rms_level > 0 else 0

        # Spectral analysis
        waveform_np = waveform[0].numpy()

        # Spectral centroid (brightness)
        fft_result = np.fft.fft(waveform_np)
        freqs = np.fft.fftfreq(len(fft_result), 1 / sample_rate)
        magnitude = np.abs(fft_result)

        valid_freqs = freqs[: len(magnitude) // 2]
        valid_magnitude = magnitude[: len(magnitude) // 2]
        spectral_centroid = np.sum(valid_freqs * valid_magnitude) / np.sum(valid_magnitude)

        # Zero crossing rate
        zero_crossing_rate = np.mean(np.diff(np.sign(waveform_np)) != 0)

        # Speech quality heuristics
        is_speech_like = self.is_speech_like(waveform_np, sample_rate)

        return {
            "basic_metrics": {
                "duration": duration,
                "rms_level": rms_level,
                "peak_level": peak_level,
                "dynamic_range_db": dynamic_range,
            },
            "spectral_metrics": {
                "spectral_centroid": spectral_centroid,
                "zero_crossing_rate": zero_crossing_rate,
            },
            "quality_scores": {
                "is_speech_like": is_speech_like,
            },
            "sample_rate": sample_rate,
            "num_samples": waveform.shape[1],
        }

    def is_speech_like(self, waveform: np.ndarray, sample_rate: int) -> bool:
        """Determine if audio is likely speech using heuristics."""
        # Zero crossing rate for speech (typically 0.05-0.15)
        zcr = np.mean(np.diff(np.sign(waveform)) != 0)

        # Spectral centroid for speech (typically 85-255 Hz for fundamental, up to 8kHz)
        fft_result = np.fft.fft(waveform)
        freqs = np.fft.fftfreq(len(fft_result), 1 / sample_rate)
        magnitude = np.abs(fft_result)

        valid_freqs = freqs[: len(magnitude) // 2]
        valid_magnitude = magnitude[: len(magnitude) // 2]
        spectral_centroid = np.sum(valid_freqs * valid_magnitude) / np.sum(valid_magnitude)

        # Speech-like criteria
        zcr_speech = 0.05 <= zcr <= 0.15
        centroid_speech = 100 <= spectral_centroid <= 4000

        return zcr_speech and centroid_speech


def create_lfm_model_manager(
    repo_id: str = "LiquidAI/LFM2.5-Audio-1.5B", device: str = None
) -> LFMModelManager:
    """Create and return an LFM model manager instance."""
    return LFMModelManager(repo_id, device)


def create_advanced_audio_processor(target_sample_rate: int = 16000) -> AdvancedAudioProcessor:
    """Create and return an advanced audio processor instance."""
    return AdvancedAudioProcessor(target_sample_rate)
