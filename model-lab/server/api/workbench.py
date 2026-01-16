from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from harness.session import SessionRunner


router = APIRouter(prefix="/api/workbench", tags=["workbench"])

_WORKER_LOCK = threading.Lock()
_WORKER_ACTIVE = False
_ACTIVE_RUN_ID: Optional[str] = None


def _runs_root() -> Path:
    return Path(os.environ.get("MODEL_LAB_RUNS_ROOT", "runs")).resolve()


def _inputs_root() -> Path:
    return Path(os.environ.get("MODEL_LAB_INPUTS_ROOT", "inputs")).resolve()


def _try_acquire_worker() -> bool:
    global _WORKER_ACTIVE
    with _WORKER_LOCK:
        if _WORKER_ACTIVE:
            return False
        _WORKER_ACTIVE = True
        return True


def _release_worker() -> None:
    global _WORKER_ACTIVE, _ACTIVE_RUN_ID
    with _WORKER_LOCK:
        _WORKER_ACTIVE = False
        _ACTIVE_RUN_ID = None


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


class RunnerBusyError(Exception):
    """Raised when runner is busy and cannot accept new runs."""
    pass


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
    if not _try_acquire_worker():
        raise RunnerBusyError("Runner is busy with another job")
    
    try:
        if steps_preset not in PRESETS:
            raise HTTPException(status_code=400, detail=f"Invalid steps_preset. Available: {list(PRESETS.keys())}")
        
        steps = PRESETS[steps_preset]["steps"]
        runner = SessionRunner(input_path, _runs_root(), steps=steps)
        
        global _ACTIVE_RUN_ID
        _ACTIVE_RUN_ID = runner.run_id
        
        now = datetime.now(timezone.utc)
        
        # Compute file info for run_request.json
        file_bytes = input_path.stat().st_size
        import hashlib
        sha256_hex = hashlib.sha256(input_path.read_bytes()).hexdigest()
        
        # Write run_request.json with experiment metadata if provided
        _write_run_request_json(
            runner.session_dir,
            requested_at=now.isoformat(),
            source="experiment" if experiment_id else "workbench",
            use_case_id=use_case_id,
            steps_preset=steps_preset,
            filename_original=input_path.name,
            content_type=None,
            bytes_uploaded=file_bytes,
            sha256_hex=sha256_hex,
            experiment_id=experiment_id,
            candidate_id=candidate_id,
            source_ref=source_ref,
            candidate_snapshot=candidate_snapshot,
        )
        
        def _bg() -> None:
            try:
                runner.run()
            finally:
                _release_worker()
        
        thread = threading.Thread(target=_bg, name=f"workbench-run-{runner.run_id}", daemon=True)
        thread.start()
        
        # Wait for the manifest to appear as RUNNING
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            if runner.manifest_path.exists():
                try:
                    m = json.loads(runner.manifest_path.read_text(encoding="utf-8"))
                    if m.get("status") == "RUNNING":
                        break
                except Exception:
                    pass
            time.sleep(0.01)
        
        return {
            "run_id": runner.run_id,
            "run_dir": str(runner.session_dir),
            "console_url": f"/runs/{runner.run_id}",
        }
    except RunnerBusyError:
        raise
    except Exception:
        _release_worker()
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


def _write_run_request_json(
    run_root: Path,
    *,
    requested_at: str,
    source: str = "workbench",
    name: Optional[str] = None,
    use_case_id: Optional[str] = None,
    model_id: Optional[str] = None,
    steps_preset: Optional[str] = None,
    steps_requested: Optional[list[str]] = None,
    filename_original: str,
    content_type: Optional[str] = None,
    bytes_uploaded: int,
    sha256_hex: str,
    experiment_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
    source_ref: Optional[Dict[str, Any]] = None,
    candidate_snapshot: Optional[Dict[str, Any]] = None,
) -> None:
    """Write run_request.json atomically at run root for reproducibility."""
    request_data = {
        "schema_version": "1",
        "requested_at": requested_at,
        "source": source,
        "name": name,
        "use_case_id": use_case_id,
        "model_id": model_id,
        "steps_preset": steps_preset,
        "steps_requested": steps_requested,
        "filename_original": filename_original,
        "content_type": content_type,
        "bytes": bytes_uploaded,
        "sha256": sha256_hex,
    }
    
    # Add experiment metadata if provided
    if experiment_id:
        request_data["experiment_id"] = experiment_id
        request_data["candidate_id"] = candidate_id
        request_data["source_ref"] = source_ref
        request_data["candidate_snapshot"] = candidate_snapshot
    
    # Atomic write
    run_root.mkdir(parents=True, exist_ok=True)
    request_path = run_root / "run_request.json"
    tmp_path = run_root / "run_request.json.tmp"
    
    tmp_path.write_text(json.dumps(request_data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, request_path)


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
    if not _try_acquire_worker():
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

        global _ACTIVE_RUN_ID
        _ACTIVE_RUN_ID = runner.run_id

        # Write run_request.json before starting runner
        _write_run_request_json(
            runner.session_dir,
            requested_at=now.isoformat(),
            source="workbench",
            use_case_id=use_case_id,
            steps_preset=steps_preset,
            filename_original=file.filename or "unknown",
            content_type=file.content_type,
            bytes_uploaded=bytes_uploaded,
            sha256_hex=sha256_hex,
        )

        def _bg() -> None:
            try:
                runner.run()
            finally:
                _release_worker()

        thread = threading.Thread(target=_bg, name=f"workbench-run-{runner.run_id}", daemon=True)
        thread.start()

        # Wait for the manifest to appear as RUNNING (critical UX contract).
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            if runner.manifest_path.exists():
                try:
                    m = json.loads(runner.manifest_path.read_text(encoding="utf-8"))
                    if m.get("status") == "RUNNING":
                        break
                except Exception:
                    pass
            time.sleep(0.01)

        if not runner.manifest_path.exists():
            raise HTTPException(status_code=500, detail="Run failed to start (manifest not created)")

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
        _release_worker()
        raise

