"""
Runner Output Schema - Canonical artifact structure for all runners.

Every runner MUST produce artifacts conforming to this schema.
Tests assert the schema, not just "some keys exist".

Usage:
    from harness.runner_schema import RunnerArtifact, validate_artifact, enforce_adhoc_metrics
    
    artifact = RunnerArtifact(
        run_context=RunContext(...),
        inputs=InputsSchema(...),
        metrics_quality=QualityMetrics(...),  # Must be None in adhoc
        metrics_structural={"rtf": 0.5, ...},
        artifacts={"output_path": "..."},
    )
    
    # Validate before writing - raises if contract violated
    validate_artifact(artifact)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path
import hashlib
import subprocess
from datetime import datetime


# Quality metrics that are FORBIDDEN in adhoc mode (must be None)
QUALITY_METRICS_FORBIDDEN = frozenset({
    "wer",              # ASR - Word Error Rate
    "cer",              # ASR - Character Error Rate
    "mer",              # ASR - Match Error Rate 
    "wil",              # ASR - Word Information Lost
    "der",              # Diarization Error Rate
    "der_proxy",        # Diarization proxy metric
    "speaker_accuracy", # Diarization speaker accuracy
    "jaccard_error",    # Diarization Jaccard error
    "bleu",             # Translation BLEU score
})


@dataclass
class RunContext:
    """Execution context for reproducibility."""
    task: str                          # asr, vad, diarization, v2v, tts
    model_id: str                      # faster_whisper, silero_vad, etc.
    grade: str                         # adhoc, smoke, golden_batch
    timestamp: str                     # ISO format
    git_hash: Optional[str]            # Runner git hash (None if not in git repo)
    command: List[str]                 # sys.argv for reproducibility
    device: str                        # cpu, mps, cuda
    model_version: Optional[str] = None


@dataclass
class InputsSchema:
    """Input provenance - tracks both source file and decoded audio."""
    audio_path: str                             # Path to decoded audio (or temp file for video)
    audio_hash: str                             # SHA256 of decoded canonical PCM (float32 mono)
    source_media_path: Optional[str] = None     # Original file path (video container, etc.)
    source_media_hash: Optional[str] = None     # SHA256 of original file bytes
    dataset_id: Optional[str] = None            # Dataset ID if from dataset
    dataset_hash: Optional[str] = None          # SHA256 of dataset YAML
    audio_duration_s: Optional[float] = None    # Duration in seconds
    sample_rate: Optional[int] = None           # Sample rate


@dataclass
class QualityMetrics:
    """
    Quality metrics requiring ground truth.
    
    ALL fields must be None when has_ground_truth=False (adhoc mode).
    This is enforced at artifact creation time.
    """
    wer: Optional[float] = None
    cer: Optional[float] = None
    mer: Optional[float] = None
    wil: Optional[float] = None
    der: Optional[float] = None
    der_proxy: Optional[float] = None
    speaker_accuracy: Optional[float] = None
    jaccard_error: Optional[float] = None
    bleu: Optional[float] = None


@dataclass 
class RunnerArtifact:
    """
    Canonical runner output artifact.
    
    Every runner produces this structure. Tests assert the schema.
    """
    run_context: RunContext
    inputs: InputsSchema
    metrics_quality: QualityMetrics
    metrics_structural: Dict[str, Any]          # Task-specific (rtf, latency_ms, etc.)
    output: Dict[str, Any]                      # Task-specific output (text, segments, etc.)
    artifacts: Dict[str, str] = field(default_factory=dict)  # output_path, logs_path, etc.
    provenance: Dict[str, Any] = field(default_factory=dict) # Legacy provenance for compatibility
    gates: Dict[str, Any] = field(default_factory=dict)      # Sanity gates
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "run_context": asdict(self.run_context),
            "inputs": asdict(self.inputs),
            "metrics_quality": asdict(self.metrics_quality),
            "metrics_structural": self.metrics_structural,
            "output": self.output,
            "artifacts": self.artifacts,
            "provenance": self.provenance,
            "gates": self.gates,
            "errors": self.errors,
            # Legacy compatibility fields
            "meta": {
                "task": self.run_context.task,
                "model_id": self.run_context.model_id,
                "timestamp": self.run_context.timestamp,
            },
            "evidence": {
                "grade": self.run_context.grade,
                "dataset_id": self.inputs.dataset_id,
            },
            "system": {
                "device": self.run_context.device,
            },
            "metrics": {
                **self.metrics_structural,
                **{k: v for k, v in asdict(self.metrics_quality).items()},
            },
        }


class ArtifactValidationError(Exception):
    """Raised when artifact violates schema contract."""
    pass


def enforce_adhoc_metrics(metrics_quality: QualityMetrics, grade: str) -> None:
    """
    Enforce that adhoc runs have no quality metrics.
    
    Raises ArtifactValidationError if any forbidden metric is not None.
    
    Args:
        metrics_quality: The quality metrics object
        grade: The evidence grade (adhoc, smoke, golden_batch)
        
    Raises:
        ArtifactValidationError: If contract violated
    """
    if grade != "adhoc":
        return  # Only enforce for adhoc
    
    violations = []
    metrics_dict = asdict(metrics_quality)
    
    for metric_name in QUALITY_METRICS_FORBIDDEN:
        if metric_name in metrics_dict and metrics_dict[metric_name] is not None:
            violations.append(f"{metric_name}={metrics_dict[metric_name]}")
    
    if violations:
        raise ArtifactValidationError(
            f"Adhoc run has forbidden quality metrics (must be None): {', '.join(violations)}"
        )


def validate_artifact(artifact: RunnerArtifact) -> None:
    """
    Validate artifact against schema contract.
    
    Raises ArtifactValidationError if contract violated.
    """
    # 1. Required fields
    if not artifact.run_context.task:
        raise ArtifactValidationError("run_context.task is required")
    if not artifact.run_context.model_id:
        raise ArtifactValidationError("run_context.model_id is required")
    if not artifact.run_context.grade:
        raise ArtifactValidationError("run_context.grade is required")
    if not artifact.inputs.audio_hash:
        raise ArtifactValidationError("inputs.audio_hash is required")
    
    # 2. Adhoc metric enforcement
    enforce_adhoc_metrics(artifact.metrics_quality, artifact.run_context.grade)
    
    # 3. Video container check - if source differs from audio, both hashes required
    if artifact.inputs.source_media_path and artifact.inputs.source_media_path != artifact.inputs.audio_path:
        if not artifact.inputs.source_media_hash:
            raise ArtifactValidationError(
                "source_media_hash required when source_media_path differs from audio_path"
            )
    
    # 4. Structural metrics must be present
    if not artifact.metrics_structural:
        raise ArtifactValidationError("metrics_structural must not be empty")


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of file bytes (first 16 hex chars)."""
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def compute_pcm_hash(audio: 'np.ndarray') -> str:
    """
    Compute SHA256 hash of decoded canonical PCM.
    
    This is the hash of the actual audio content, not the file bytes.
    Ensures provenance is robust across re-encodes.
    
    Args:
        audio: numpy array of audio samples (float32)
        
    Returns:
        First 16 hex chars of SHA256
    """
    import numpy as np
    # Ensure float32 for consistency
    audio_bytes = audio.astype(np.float32).tobytes()
    return hashlib.sha256(audio_bytes).hexdigest()[:16]


def get_git_hash() -> Optional[str]:
    """Get current git hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def create_run_context(
    task: str,
    model_id: str,
    grade: str,
    device: str,
    command: Optional[List[str]] = None,
    model_version: Optional[str] = None,
) -> RunContext:
    """
    Create RunContext with automatic timestamp and git hash.
    
    Args:
        task: Task type (asr, vad, diarization, v2v, tts)
        model_id: Model identifier
        grade: Evidence grade (adhoc, smoke, golden_batch)
        device: Compute device (cpu, mps, cuda)
        command: sys.argv (defaults to current)
        model_version: Optional model version string
    """
    import sys
    return RunContext(
        task=task,
        model_id=model_id,
        grade=grade,
        timestamp=datetime.now().isoformat(),
        git_hash=get_git_hash(),
        command=command or sys.argv,
        device=device,
        model_version=model_version,
    )


def create_inputs_schema(
    audio_path: Path,
    audio_array: 'np.ndarray',
    source_media_path: Optional[Path] = None,
    dataset_id: Optional[str] = None,
    dataset_path: Optional[Path] = None,
    audio_duration_s: Optional[float] = None,
    sample_rate: Optional[int] = None,
) -> InputsSchema:
    """
    Create InputsSchema with proper hashing.
    
    Args:
        audio_path: Path to audio file (may be temp file for video)
        audio_array: Decoded audio as numpy array
        source_media_path: Original media file (if different from audio_path)
        dataset_id: Dataset identifier
        dataset_path: Path to dataset YAML
        audio_duration_s: Audio duration
        sample_rate: Sample rate
    """
    source = source_media_path or audio_path
    
    return InputsSchema(
        audio_path=str(audio_path),
        audio_hash=compute_pcm_hash(audio_array),
        source_media_path=str(source) if source != audio_path else None,
        source_media_hash=compute_file_hash(source) if source != audio_path else None,
        dataset_id=dataset_id,
        dataset_hash=compute_file_hash(dataset_path) if dataset_path and dataset_path.exists() else None,
        audio_duration_s=audio_duration_s,
        sample_rate=sample_rate,
    )
