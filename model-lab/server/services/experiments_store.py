"""
Experiments storage layer.

Manages experiment lifecycle on disk without touching harness/session.py.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def _experiments_root() -> Path:
    """Get experiments directory under runs root."""
    runs_root = Path(os.environ.get("MODEL_LAB_RUNS_ROOT", "runs")).resolve()
    return runs_root / "experiments"


def _generate_experiment_id() -> str:
    """Generate deterministic experiment ID: exp_{YYYYMMDD_HHMMSS}_{hex}"""
    now = datetime.now(UTC)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    random_hex = uuid4().hex[:8]
    return f"exp_{timestamp}_{random_hex}"


def create_experiment(
    *,
    use_case_id: str,
    filename_original: str,
    content_type: str | None,
    bytes_uploaded: int,
    sha256_hex: str,
    input_rel_path: str,
    candidates: list[dict[str, str]],
) -> dict[str, Any]:
    """
    Create experiment directory and write request/state files.

    Returns experiment_request data.
    """
    experiment_id = _generate_experiment_id()
    exp_dir = _experiments_root() / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(UTC)

    # Build experiment_request.json
    request_data = {
        "schema_version": "1",
        "experiment_id": experiment_id,
        "created_at": now.isoformat(),
        "use_case_id": use_case_id,
        "source": {
            "filename_original": filename_original,
            "content_type": content_type,
            "bytes": bytes_uploaded,
            "sha256": sha256_hex,
            "rel_path": input_rel_path,
        },
        "candidates": candidates,
    }

    # Atomic write request
    request_path = exp_dir / "experiment_request.json"
    tmp_path = exp_dir / "experiment_request.json.tmp"
    tmp_path.write_text(json.dumps(request_data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, request_path)

    # Initialize experiment_state.json
    state_data = {
        "schema_version": "1",
        "experiment_id": experiment_id,
        "runs": [
            {
                "candidate_id": c["candidate_id"],
                "steps_preset": c["steps_preset"],
                "run_id": None,
                "status": "QUEUED",
                "created_at": now.isoformat(),
                "started_at": None,
                "ended_at": None,
            }
            for c in candidates
        ],
        "last_updated_at": now.isoformat(),
    }

    # Atomic write state
    state_path = exp_dir / "experiment_state.json"
    tmp_state = exp_dir / "experiment_state.json.tmp"
    tmp_state.write_text(json.dumps(state_data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_state, state_path)

    return request_data


def get_experiment(experiment_id: str) -> dict[str, Any] | None:
    """Load experiment request + state, return merged summary."""
    exp_dir = _experiments_root() / experiment_id
    if not exp_dir.exists():
        return None

    request_path = exp_dir / "experiment_request.json"
    state_path = exp_dir / "experiment_state.json"

    if not request_path.exists() or not state_path.exists():
        return None

    try:
        request_data = json.loads(request_path.read_text(encoding="utf-8"))
        state_data = json.loads(state_path.read_text(encoding="utf-8"))

        # Merge for UI convenience
        return {
            **request_data,
            "runs": state_data["runs"],
            "last_updated_at": state_data["last_updated_at"],
        }
    except Exception:
        return None


def update_experiment_state(experiment_id: str, runs: list[dict[str, Any]]) -> None:
    """Update experiment_state.json atomically."""
    exp_dir = _experiments_root() / experiment_id
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment {experiment_id} not found")

    state_path = exp_dir / "experiment_state.json"

    # Read existing to preserve schema_version
    existing = json.loads(state_path.read_text(encoding="utf-8"))

    updated = {
        "schema_version": existing["schema_version"],
        "experiment_id": experiment_id,
        "runs": runs,
        "last_updated_at": datetime.now(UTC).isoformat(),
    }

    tmp_path = exp_dir / "experiment_state.json.tmp"
    tmp_path.write_text(json.dumps(updated, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, state_path)


def get_next_queued_run(experiment_id: str) -> dict[str, Any] | None:
    """Find first QUEUED run slot."""
    exp = get_experiment(experiment_id)
    if not exp:
        return None

    for run in exp["runs"]:
        if run["status"] == "QUEUED":
            return run

    return None
