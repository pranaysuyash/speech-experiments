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


async def _save_upload_to_disk(upload: UploadFile, dest: Path, max_bytes: int) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with dest.open("wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(status_code=413, detail="Upload too large")
            f.write(chunk)
    return total


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
        await _save_upload_to_disk(file, dest, max_upload)

        if steps_preset not in {"full", "ingest"}:
            raise HTTPException(status_code=400, detail="Invalid steps_preset")

        steps = ["ingest"] if steps_preset == "ingest" else None
        runner = SessionRunner(dest, _runs_root(), steps=steps)

        global _ACTIVE_RUN_ID
        _ACTIVE_RUN_ID = runner.run_id

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

