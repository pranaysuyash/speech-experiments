"""
Adhoc dataset helper for single-file runs.

Creates dataset metadata for adhoc (user-provided file) execution.
Adhoc runs produce artifacts but don't contribute to decision evidence.
"""

import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AdhocDataset:
    """Metadata for an adhoc (user-provided) audio file."""
    dataset_id: str        # adhoc_<audio_hash[:12]>
    audio_hash: str        # sha256 of file bytes
    dataset_hash: str      # same as audio_hash for adhoc
    grade: str             # always "adhoc"
    has_ground_truth: bool # always False
    audio_path: str        # absolute path to file
    audio_duration_s: float
    sample_rate: int
    channels: int


def create_adhoc_dataset(
    audio_path: Path,
    task: Optional[str] = None,
    prompt: Optional[str] = None,
) -> AdhocDataset:
    """
    Create adhoc dataset metadata from a single audio file.
    
    Args:
        audio_path: Path to audio file
        task: Optional task name (for logging)
        prompt: Optional prompt (for v2v/tts tasks)
    
    Returns:
        AdhocDataset with all required provenance fields
    """
    import wave
    import struct
    
    audio_path = Path(audio_path).resolve()
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Compute file hash
    with open(audio_path, 'rb') as f:
        file_bytes = f.read()
    audio_hash = hashlib.sha256(file_bytes).hexdigest()
    
    # Create stable dataset_id from hash
    dataset_id = f"adhoc_{audio_hash[:12]}"
    
    # Try to get audio properties
    duration_s = 0.0
    sample_rate = 0
    channels = 1
    
    try:
        # Try librosa first (most flexible)
        import librosa
        y, sr = librosa.load(str(audio_path), sr=None, mono=False)
        sample_rate = sr
        if len(y.shape) == 1:
            channels = 1
            duration_s = len(y) / sr
        else:
            channels = y.shape[0]
            duration_s = y.shape[1] / sr
    except ImportError:
        # Fallback to wave for WAV files
        try:
            with wave.open(str(audio_path), 'rb') as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                duration_s = wf.getnframes() / sample_rate
        except Exception:
            # Can't determine properties
            pass
    except Exception:
        pass
    
    return AdhocDataset(
        dataset_id=dataset_id,
        audio_hash=audio_hash,
        dataset_hash=audio_hash,  # Same for adhoc
        grade="adhoc",
        has_ground_truth=False,
        audio_path=str(audio_path),
        audio_duration_s=duration_s,
        sample_rate=sample_rate,
        channels=channels,
    )


def create_adhoc_provenance(adhoc: AdhocDataset) -> Dict[str, Any]:
    """Create provenance dict for adhoc run."""
    return {
        "has_ground_truth": False,
        "dataset_hash": adhoc.dataset_hash,
        "audio_hash": adhoc.audio_hash,
        "metrics_valid": True,  # Structural metrics only
        "quality_metrics_allowed": False,  # WER, DER etc forbidden
    }


def create_adhoc_run_context(
    adhoc: AdhocDataset,
    device: str,
    runner_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Create run_context dict for adhoc run."""
    import subprocess
    
    # Get git hash
    git_hash = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            git_hash = result.stdout.strip()
    except Exception:
        pass
    
    return {
        "device": device,
        "audio_duration_s": adhoc.audio_duration_s,
        "sample_rate": adhoc.sample_rate,
        "channels": adhoc.channels,
        "git_hash": git_hash,
        "runner_version": runner_version or "1.0.0",
        "grade": "adhoc",
    }


# Quality metrics that are FORBIDDEN in adhoc mode
QUALITY_METRICS_FORBIDDEN = {
    "wer",           # ASR
    "cer",           # ASR
    "mer",           # ASR
    "wil",           # ASR
    "der",           # Diarization
    "der_proxy",     # Diarization
    "speaker_accuracy",  # Diarization
    "jaccard_error", # Diarization
}


def validate_adhoc_metrics(metrics: Dict[str, Any]) -> bool:
    """
    Validate that adhoc metrics don't contain quality metrics.
    
    Returns True if valid (no forbidden metrics), False otherwise.
    """
    for key in metrics:
        if key.lower() in QUALITY_METRICS_FORBIDDEN:
            if metrics[key] is not None:
                return False
    return True
