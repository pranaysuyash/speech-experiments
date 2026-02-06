"""
Run Provenance - Fields and utilities for machine-verifiable run artifacts.

Every run artifact should include these fields to enable:
- Stale detection (dataset_hash mismatch)
- No-GT detection (has_ground_truth=false means quality metrics invalid)
- Poisoned run detection (metrics_valid=false)

Usage in run_*.py scripts:
    from harness.run_provenance import create_provenance, RUN_SCHEMA_VERSION

    provenance = create_provenance(
        dataset_id="asr_golden_v1",
        dataset_path=Path("data/golden/asr_golden_v1.yaml"),
        audio_path=Path("data/audio/clean_speech_10s.wav"),
        ground_truth_path=Path("data/audio/clean_speech_10s.txt"),  # or None
    )

    run_artifact = {
        "provenance": provenance,
        "metrics": {...},
        ...
    }
"""

import hashlib
from datetime import datetime
from pathlib import Path

RUN_SCHEMA_VERSION = "1.0"


def compute_file_hash(path: Path, algorithm: str = "sha256") -> str:
    """Compute hash of file contents. Returns first 16 chars of hex digest."""
    if not path.exists():
        return "file_not_found"
    with open(path, "rb") as f:
        digest = hashlib.new(algorithm, f.read()).hexdigest()
    return digest[:16]


def create_provenance(
    dataset_id: str,
    dataset_path: Path | None = None,
    audio_path: Path | None = None,
    ground_truth_path: Path | None = None,
    metrics_valid: bool = True,
    invalid_reason: str | None = None,
) -> dict:
    """
    Create provenance fields for a run artifact.

    Args:
        dataset_id: Identifier for the dataset (e.g., "asr_golden_v1")
        dataset_path: Path to dataset definition file (YAML)
        audio_path: Path to primary audio file
        ground_truth_path: Path to ground truth text/RTTM (None if not available)
        metrics_valid: Whether quality metrics can be trusted
        invalid_reason: Reason if metrics_valid=False

    Returns:
        Dict with all provenance fields
    """
    provenance = {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(),
        "dataset_id": dataset_id,
        "has_ground_truth": ground_truth_path is not None and ground_truth_path.exists(),
        "metrics_valid": metrics_valid,
    }

    # Dataset hash (for staleness detection)
    if dataset_path and dataset_path.exists():
        provenance["dataset_hash"] = compute_file_hash(dataset_path)
    else:
        provenance["dataset_hash"] = None

    # Audio hash
    if audio_path and audio_path.exists():
        provenance["audio_hash"] = compute_file_hash(audio_path)
    else:
        provenance["audio_hash"] = None

    # Ground truth hash (only if exists)
    if ground_truth_path and ground_truth_path.exists():
        provenance["ground_truth_hash"] = compute_file_hash(ground_truth_path)
    else:
        provenance["ground_truth_hash"] = None

    # Invalid reason
    if not metrics_valid and invalid_reason:
        provenance["invalid_reason"] = invalid_reason

    return provenance


def validate_provenance(provenance: dict) -> tuple:
    """
    Validate provenance fields.

    Returns (is_valid, issues)
    """
    issues = []

    required = ["schema_version", "dataset_id", "has_ground_truth", "metrics_valid"]
    for field in required:
        if field not in provenance:
            issues.append(f"Missing required field: {field}")

    return len(issues) == 0, issues


def is_stale(provenance: dict, current_dataset_hash: str) -> bool:
    """Check if run is stale (dataset definition changed)."""
    stored_hash = provenance.get("dataset_hash")
    if stored_hash is None:
        return False  # Can't determine staleness without hash
    return stored_hash != current_dataset_hash


def can_compute_quality_metrics(provenance: dict) -> bool:
    """Check if quality metrics (WER, speaker accuracy) can be computed."""
    return provenance.get("has_ground_truth", False) and provenance.get("metrics_valid", True)


def create_run_context(
    device: str,
    audio_duration_s: float | None = None,
    model_version: str | None = None,
    runner_git_hash: str | None = None,
) -> dict:
    """
    Create run_context fields for interpretable latency metrics.

    Latency numbers are meaningless without context. This records:
    - device (cpu/mps/cuda)
    - audio_duration_s (for RTF calculation)
    - model_version (for reproducibility)
    - runner_git_hash (for debugging)

    Usage:
        run_context = create_run_context(
            device="mps",
            audio_duration_s=10.0,
            model_version="v3.0.0",
        )
        run_artifact["run_context"] = run_context
    """
    import subprocess

    # Try to get git hash if not provided
    if runner_git_hash is None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                runner_git_hash = result.stdout.strip()
        except:
            pass

    return {
        "device": device,
        "audio_duration_s": audio_duration_s,
        "model_version": model_version,
        "runner_git_hash": runner_git_hash,
    }
