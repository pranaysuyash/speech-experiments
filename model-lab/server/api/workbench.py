from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from harness.session import SessionRunner
from server.services.lifecycle import (
    try_acquire_worker,
    release_worker,
    launch_run_worker,
    RunnerBusyError,
)

router = APIRouter(prefix="/api/workbench", tags=["workbench"])

def _runs_root() -> Path:
    return Path(os.environ.get("MODEL_LAB_RUNS_ROOT", "runs")).resolve()


def _inputs_root() -> Path:
    return Path(os.environ.get("MODEL_LAB_INPUTS_ROOT", "inputs")).resolve()


def _safe_filename(name: str) -> str:
    # Keep this conservative; storage is for provenance, not pretty names.
    name = (name or "").strip()
    if not name:
        return "upload"
    name = name.replace("\\", "_").replace("/", "_")
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)[:120] or "upload"


# Preset registry - source of truth for available step presets
PRESETS = {
    "ingest": {
        "label": "Ingest Only",
        "description": "Fast preprocessing: normalize audio and create bundle structure",
        "steps": ["ingest"]
    },
    "full": {
        "label": "Full Pipeline",
        "description": "Complete pipeline: ASR, diarization, alignment, and summary generation",
        "steps": None  # None means all steps
    }
}


@router.get("/presets")
def get_presets() -> list[dict]:
    """
    Return available step presets.
    
    This is the source of truth for what presets experiments can use.
    """
    return [
        {
            "steps_preset": key,
            "label": meta["label"],
            "description": meta.get("description")
        }
        for key, meta in PRESETS.items()
    ]


def start_run_from_path(
    input_path: Path,
    use_case_id: str,
    steps_preset: str,
    *,
    experiment_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
    source_ref: Optional[Dict[str, Any]] = None,
    candidate_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Start a run from a file path (not multipart upload).
    
    This is the pure function callable by experiments API.
    
    Args:
        input_path: Path to input media file
        use_case_id: Use case identifier
        steps_preset: Preset name (must exist in PRESETS)
        experiment_id: Optional experiment this run belongs to
        candidate_id: Optional candidate ID within experiment
        source_ref: Optional source reference metadata
    
    Returns:
        {run_id, run_dir, console_url}
    
    Raises:
        RunnerBusyError: If runner is already processing another job
        HTTPException: If preset is invalid
    """
    if not try_acquire_worker():
        raise RunnerBusyError("Runner is busy with another job")
    
    try:
        if steps_preset not in PRESETS:
            raise HTTPException(status_code=400, detail=f"Invalid steps_preset. Available: {list(PRESETS.keys())}")
        
        steps = PRESETS[steps_preset]["steps"]
        runner = SessionRunner(input_path, _runs_root(), steps=steps)
        
        now = datetime.now(timezone.utc)
        
        # Compute file info for run_request.json
        file_bytes = input_path.stat().st_size
        import hashlib
        sha256_hex = hashlib.sha256(input_path.read_bytes()).hexdigest()
        
        # Construct run_request data
        run_request_data = {
            "schema_version": "1",
            "requested_at": now.isoformat(),
            "source": "experiment" if experiment_id else "workbench",
            "use_case_id": use_case_id,
            "steps_preset": steps_preset,
            "steps_requested": steps,
            "filename_original": input_path.name,
            "content_type": None,
            "bytes_uploaded": file_bytes,
            "sha256": sha256_hex,
        }
        
        if experiment_id:
            run_request_data["experiment_id"] = experiment_id
            run_request_data["candidate_id"] = candidate_id
            run_request_data["source_ref"] = source_ref
            run_request_data["candidate_snapshot"] = candidate_snapshot

        # Launch worker
        result = launch_run_worker(runner, run_request_data, background=True)
        
        return {
            "run_id": runner.run_id,
            "input_hash": sha256_hex,
            "run_dir": str(runner.session_dir),
            "console_url": f"/runs/{runner.run_id}",
            "worker_pid": result.get("worker_pid"),
        }
    except Exception:
        release_worker()
        raise


async def _save_upload_to_disk(upload: UploadFile, dest: Path, max_bytes: int) -> tuple[int, str]:
    """
    Save upload to disk while computing sha256.
    
    Returns (bytes_written, sha256_hex).
    """
    import hashlib
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    h = hashlib.sha256()
    
    with dest.open("wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(status_code=413, detail="Upload too large")
            h.update(chunk)
            f.write(chunk)
    
    return total, h.hexdigest()


@router.post("/runs")
async def create_workbench_run(
    file: UploadFile = File(...),
    use_case_id: str = Form(...),
    steps_preset: str = Form("full"),
) -> JSONResponse:
    """
    Create a run from the UI (multipart upload) and start processing asynchronously.

    Contract:
    - Returns quickly once run dir exists and manifest is RUNNING.
    - Single-worker guardrail: returns 409 RUNNER_BUSY if a run is already active.
    - Writes run_request.json at run root for reproducibility.
    """
    if not try_acquire_worker():
        return JSONResponse(status_code=409, content={"error_code": "RUNNER_BUSY"})

    try:
        max_upload = int(os.environ.get("MODEL_LAB_WORKBENCH_MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))
        now = datetime.now(timezone.utc)
        yyyy_mm = now.strftime("%Y-%m")
        inputs_dir = _inputs_root() / "workbench" / yyyy_mm
        filename = _safe_filename(file.filename or "upload")
        dest = inputs_dir / f"{uuid4().hex}_{filename}"
        
        # Save upload and compute sha256
        bytes_uploaded, sha256_hex = await _save_upload_to_disk(file, dest, max_upload)

        if steps_preset not in PRESETS:
            raise HTTPException(status_code=400, detail=f"Invalid steps_preset. Available: {list(PRESETS.keys())}")
        
        steps = PRESETS[steps_preset]["steps"]
        runner = SessionRunner(dest, _runs_root(), steps=steps)

        # Construct run_request data
        run_request_data = {
            "schema_version": "1",
            "requested_at": now.isoformat(),
            "source": "workbench",
            "use_case_id": use_case_id,
            "steps_preset": steps_preset,
            "steps_requested": steps,
            "filename_original": file.filename or "unknown",
            "content_type": file.content_type,
            "bytes_uploaded": bytes_uploaded,
            "sha256": sha256_hex,
        }
        
        # Launch worker
        launch_run_worker(runner, run_request_data, background=True)

        # Write UI metadata without mutating the run manifest (multi-agent safe).
        try:
            meta: Dict[str, Any] = {
                "use_case_id": use_case_id,
                "steps_preset": steps_preset,
                "input_rel_path": str(dest.relative_to(_inputs_root())) if dest.is_relative_to(_inputs_root()) else str(dest),
                "created_at": now.isoformat(),
            }
            (runner.session_dir / "workbench.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            # Best-effort only.
            pass

        return JSONResponse(
            status_code=200,
            content={
                "run_id": runner.run_id,
                "run_dir": str(runner.session_dir),
                "console_url": f"/runs/{runner.run_id}",
            },
        )
    except Exception:
        release_worker()
        raise
