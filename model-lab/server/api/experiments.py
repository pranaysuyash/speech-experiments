"""
Experiments API - Compare runs from a single upload.

Endpoints:
- POST /api/experiments - Create experiment with 2 candidates
- GET /api/experiments/{id} - Get experiment summary
- POST /api/experiments/{id}/runs/start - Start next QUEUED candidate
- POST /api/experiments/{id}/runs/start-all - Start one, return summary
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from server.api.workbench import PRESETS, RunnerBusyError, start_run_from_path


router = APIRouter(prefix="/api/experiments", tags=["experiments"])


def _runs_root() -> Path:
    return Path(os.environ.get("MODEL_LAB_RUNS_ROOT", "runs")).resolve()


def _experiments_root() -> Path:
    return _runs_root() / "experiments"


def _generate_experiment_id() -> str:
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    random_hex = uuid4().hex[:8]
    return f"exp_{timestamp}_{random_hex}"


def _safe_filename(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return "upload"
    name = name.replace("\\", "_").replace("/", "_")
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)[:120] or "upload"


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Atomic JSON write via tmp + replace."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _load_experiment(experiment_id: str) -> Optional[Dict[str, Any]]:
    """Load experiment request + state merged."""
    exp_dir = _experiments_root() / experiment_id
    request_path = exp_dir / "experiment_request.json"
    state_path = exp_dir / "experiment_state.json"
    
    if not request_path.exists() or not state_path.exists():
        return None
    
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        state = json.loads(state_path.read_text(encoding="utf-8"))
        
        runs = state["runs"]
        # Enrich runs with eval data if available
        from server.api.eval_loader import load_eval
        for run in runs:
            rid = run.get("run_id")
            if rid:
                 run_path = _runs_root() / rid
                 eval_data = load_eval(run_path)
                 if eval_data:
                     run["score_cards"] = eval_data.get("score_cards", [])
        
        return {**request, "runs": runs, "last_updated_at": state["last_updated_at"]}
    except Exception:
        return None


@router.post("")
async def create_experiment(
    file: UploadFile = File(...),
    use_case_id: str = Form(...),
    candidate_ids: Optional[str] = Form(None),  # comma-separated
) -> JSONResponse:
    """
    Create an experiment with 2 candidate runs.
    
    Accepts either:
    - candidate_ids: comma-separated candidate IDs (preferred)
    - Falls back to first 2 candidates for use_case_id
    
    Returns 400 if fewer than 2 candidates exist.
    """
    from server.api.candidates import (
        get_candidates_for_use_case,
        get_candidate,
        get_candidate_snapshot,
        USE_CASES,
    )
    
    # Parse candidate_ids if provided
    parsed_candidate_ids = None
    if candidate_ids:
        parsed_candidate_ids = [c.strip() for c in candidate_ids.split(",") if c.strip()]
    
    # Validate and get candidates
    if parsed_candidate_ids and len(parsed_candidate_ids) >= 2:
        # User specified candidates
        candidates_list = []
        for cid in parsed_candidate_ids[:2]:
            c = get_candidate(cid)
            if not c:
                return JSONResponse(
                    status_code=400,
                    content={"error_code": "INVALID_CANDIDATE", "message": f"Candidate not found: {cid}"}
                )
            if c.use_case_id != use_case_id:
                return JSONResponse(
                    status_code=400,
                    content={"error_code": "CANDIDATE_USE_CASE_MISMATCH", "message": f"Candidate {cid} is for {c.use_case_id}, not {use_case_id}"}
                )
            candidates_list.append(c)
    else:
        # Fall back to use_case's candidates
        candidates_list = get_candidates_for_use_case(use_case_id)
        if len(candidates_list) < 2:
            # Fall back to presets-based candidates
            preset_keys = list(PRESETS.keys())
            if len(preset_keys) < 2:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error_code": "NEED_TWO_CANDIDATES",
                        "message": f"Experiments require at least 2 candidates. Use case {use_case_id} has {len(candidates_list)}."
                    }
                )
            # Create synthetic candidates from presets
            from dataclasses import dataclass
            @dataclass
            class SyntheticCandidate:
                candidate_id: str
                label: str
                use_case_id: str
                steps_preset: str
                params: dict
            candidates_list = [
                SyntheticCandidate(
                    candidate_id=f"preset_{preset_keys[0]}",
                    label=PRESETS[preset_keys[0]]["label"],
                    use_case_id=use_case_id,
                    steps_preset=preset_keys[0],
                    params={},
                ),
                SyntheticCandidate(
                    candidate_id=f"preset_{preset_keys[1]}",
                    label=PRESETS[preset_keys[1]]["label"],
                    use_case_id=use_case_id,
                    steps_preset=preset_keys[1],
                    params={},
                ),
            ]
    
    experiment_id = _generate_experiment_id()
    exp_dir = _experiments_root() / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save input file
    input_dir = exp_dir / "input"
    input_dir.mkdir(exist_ok=True)
    
    filename = _safe_filename(file.filename or "upload")
    input_path = input_dir / filename
    
    # Stream file and compute sha256
    file_bytes = 0
    h = hashlib.sha256()
    with input_path.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            file_bytes += len(chunk)
            f.write(chunk)
    
    sha256_hex = h.hexdigest()
    now = datetime.now(timezone.utc)
    
    # Build candidates with snapshots (for reproducibility)
    candidate_configs = []
    for i, c in enumerate(candidates_list[:2]):
        config = {
            "candidate_id": ["A", "B"][i],
            "candidate_ref": c.candidate_id,  # original candidate ID
            "label": c.label,
            "steps_preset": c.steps_preset,
            "candidate_snapshot": {
                "candidate_id": c.candidate_id,
                "label": c.label,
                "steps_preset": c.steps_preset,
                "params": c.params if hasattr(c, 'params') else {},
            }
        }
        candidate_configs.append(config)
    
    # Write experiment_request.json
    request_data = {
        "schema_version": "2",  # upgraded from 1 to include candidate snapshots
        "experiment_id": experiment_id,
        "created_at": now.isoformat(),
        "use_case_id": use_case_id,
        "source": {
            "filename_original": file.filename or filename,
            "content_type": file.content_type,
            "bytes": file_bytes,
            "sha256": sha256_hex,
            "rel_path": f"input/{filename}",
        },
        "candidates": candidate_configs,
    }
    _atomic_write_json(exp_dir / "experiment_request.json", request_data)
    
    # Write experiment_state.json
    state_data = {
        "schema_version": "1",
        "experiment_id": experiment_id,
        "runs": [
            {
                "candidate_id": c["candidate_id"],
                "candidate_ref": c["candidate_ref"],
                "steps_preset": c["steps_preset"],
                "run_id": None,
                "status": "QUEUED",
                "created_at": now.isoformat(),
                "started_at": None,
                "ended_at": None,
            }
            for c in candidate_configs
        ],
        "last_updated_at": now.isoformat(),
    }
    _atomic_write_json(exp_dir / "experiment_state.json", state_data)
    
    return JSONResponse(
        status_code=201,
        content={
            "experiment_id": experiment_id,
            "candidates": candidate_configs,
            "source": request_data["source"],
            "experiment_url": f"/lab/experiments/{experiment_id}",
        }
    )


@router.get("/{experiment_id}")
def get_experiment(experiment_id: str) -> JSONResponse:
    """Get experiment summary (merged request + state)."""
    exp = _load_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return JSONResponse(content=exp)


@router.post("/{experiment_id}/runs/start")
def start_next_run(experiment_id: str) -> JSONResponse:
    """
    Start the next QUEUED candidate.
    
    Returns 409 if runner busy, 404 if experiment not found.
    """
    exp_dir = _experiments_root() / experiment_id
    state_path = exp_dir / "experiment_state.json"
    
    if not state_path.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    state = json.loads(state_path.read_text(encoding="utf-8"))
    request_path = exp_dir / "experiment_request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    
    # Find first QUEUED slot
    queued_slot = None
    queued_idx = -1
    for i, run in enumerate(state["runs"]):
        if run["status"] == "QUEUED":
            queued_slot = run
            queued_idx = i
            break
    
    if not queued_slot:
        return JSONResponse(
            status_code=200,
            content={"started": False, "message": "No QUEUED candidates"}
        )
    
    # Get input path
    input_rel = request["source"]["rel_path"]
    input_path = exp_dir / input_rel
    
    if not input_path.exists():
        raise HTTPException(status_code=500, detail="Experiment input file missing")
    
    # Get candidate snapshot from experiment request
    candidate_config = next(
        (c for c in request["candidates"] if c["candidate_id"] == queued_slot["candidate_id"]),
        None
    )
    candidate_snapshot = candidate_config.get("candidate_snapshot") if candidate_config else None
    
    # Try to start run
    try:
        result = start_run_from_path(
            input_path,
            request["use_case_id"],
            queued_slot["steps_preset"],
            experiment_id=experiment_id,
            candidate_id=queued_slot["candidate_id"],
            source_ref={
                "kind": "experiment_input",
                "experiment_id": experiment_id,
                "rel_path": f"experiments/{experiment_id}/{input_rel}",
            },
            candidate_snapshot=candidate_snapshot,
        )
    except RunnerBusyError:
        return JSONResponse(status_code=409, content={"error_code": "RUNNER_BUSY"})
    
    # Update state
    now = datetime.now(timezone.utc)
    state["runs"][queued_idx]["status"] = "RUNNING"
    state["runs"][queued_idx]["run_id"] = result["run_id"]
    state["runs"][queued_idx]["started_at"] = now.isoformat()
    state["last_updated_at"] = now.isoformat()
    
    _atomic_write_json(state_path, state)
    
    return JSONResponse(
        status_code=200,
        content={
            "started": True,
            "candidate_id": queued_slot["candidate_id"],
            "run_id": result["run_id"],
            "console_url": result["console_url"],
        }
    )


@router.post("/{experiment_id}/runs/start-all")
def start_all_runs(experiment_id: str) -> JSONResponse:
    """
    Attempt to start one candidate (best effort), return full summary.
    """
    # Try to start one
    start_next_run(experiment_id)  # Best effort, ignore result
    
    # Return current state
    exp = _load_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return JSONResponse(content={"experiment": exp})


@router.get("/{experiment_id}/compare")
def compare_experiment_runs(
    experiment_id: str,
    left: str,
    right: str,
    artifact: str,
    max_bytes: int = 200_000,
) -> JSONResponse:
    """
    Compare two runs in an experiment.
    
    Artifacts supported: transcript, summary, action_items.
    Enforces size limit for diff sanity.
    """
    allowed_artifacts = ["transcript", "summary", "action_items"]
    if artifact not in allowed_artifacts:
        raise HTTPException(status_code=400, detail=f"Invalid artifact. Allowed: {allowed_artifacts}")
        
    exp = _load_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
        
    runs_map = {r["run_id"]: r for r in exp["runs"] if r["run_id"]}
    
    if left not in runs_map or right not in runs_map:
        raise HTTPException(status_code=400, detail="Run IDs must belong to this experiment")
        
    def _read_artifact(run_id: str) -> Dict[str, Any]:
        """Read artifact safely with fallback logic."""
        if not run_id:
            return {"available": False, "text": None}

        # Resolution rules: Bundle first, then legacy/task paths
        run_dir = _runs_root() / run_id
        paths = []
        
        if artifact == "transcript":
            paths = [
                run_dir / "bundle" / "transcript.txt",
                run_dir / "asr" / "transcript.txt",
            ]
        elif artifact == "summary":
            paths = [
                run_dir / "bundle" / "summary.md",
                run_dir / "session" / "summary.md",
            ]
        elif artifact == "action_items":
            paths = [
                run_dir / "bundle" / "action_items.csv",
                run_dir / "session" / "action_items.csv",
            ]
            
        target_path = next((p for p in paths if p.exists()), None)
        
        if not target_path:
            return {"available": False, "text": None}
            
        file_size = target_path.stat().st_size
        if file_size > max_bytes:
            return {
                "available": True, 
                "text": None, 
                "truncated": True, 
                "size": file_size,
                "error": "PREVIEW_TOO_LARGE"
            }
            
        try:
            text = target_path.read_text(encoding="utf-8")
            return {
                "available": True,
                "text": text,
                "size": len(text)
            }
        except Exception:
            return {"available": False, "text": None, "error": "READ_ERROR"}

    left_data = _read_artifact(left)
    right_data = _read_artifact(right)
    
    # If both requested files form a "too large" pair, signal 413
    if left_data.get("error") == "PREVIEW_TOO_LARGE" or right_data.get("error") == "PREVIEW_TOO_LARGE":
         return JSONResponse(
            status_code=413,
            content={
                "error_code": "PREVIEW_TOO_LARGE",
                "left": left_data,
                "right": right_data
            }
        )

    return JSONResponse(
        content={
            "artifact": artifact,
            "left": left_data,
            "right": right_data,
        }
    )
