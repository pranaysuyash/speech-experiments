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
from server.api.candidates import (
    get_candidates_for_use_case,
    get_candidate,
    get_candidate_snapshot,
    USE_CASES,
)
from server.services.compare_results_v1 import compute_comparison_v1
from server.services.results_v1 import compute_result_v1


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
        
        # Provenance Check
        prov_status = "UNVERIFIED"
        stored_prov = state.get("provenance")
        
        if stored_prov and stored_prov.get("algorithm") == "sha256":
            req_canonical = json.dumps(request, sort_keys=True, separators=(',', ':')).encode('utf-8')
            req_hash = hashlib.sha256(req_canonical).hexdigest()
            
            if req_hash == stored_prov.get("hash"):
                prov_status = "VERIFIED"
            else:
                prov_status = "CORRUPTED"
                import logging
                logging.getLogger("server.experiments").error(
                    f"Provenance Mismatch for {experiment_id}: stored={stored_prov.get('hash')}, computed={req_hash}"
                )
        
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
        
        return {**request, "runs": runs, "last_updated_at": state["last_updated_at"], "provenance_status": prov_status}
    except Exception:
        return None


@router.post("")
async def create_experiment(
    file: UploadFile = File(...),
    use_case_id: str = Form(...),
    candidate_ids: Optional[str] = Form(None),  # comma-separated, optional
    config: Optional[str] = Form(None),  # JSON config overrides
) -> JSONResponse:
    
    # Parse candidate_ids if provided
    parsed_candidate_ids = None
    if candidate_ids:
         parsed_candidate_ids = [c.strip() for c in candidate_ids.split(",") if c.strip()]
    
    # Validate and get candidates
    if parsed_candidate_ids and len(parsed_candidate_ids) >= 1:
        # User specified candidates
        candidates_list = []
        for cid in parsed_candidate_ids:
            c = get_candidate(cid)
            if not c:
                return JSONResponse(
                    status_code=400,
                    content={"error_code": "INVALID_CANDIDATE", "error_message": f"Candidate not found: {cid}"}
                )
            if c.use_case_id != use_case_id:
                return JSONResponse(
                    status_code=400,
                    content={"error_code": "CANDIDATE_USE_CASE_MISMATCH", "error_message": f"Candidate {cid} is for {c.use_case_id}, not {use_case_id}"}
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
                        "error_code": "INVALID_CANDIDATE_COUNT",
                        "error_message": f"Experiments require at least 2 candidates. Use case {use_case_id} has {len(candidates_list)}."
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
    for i, c in enumerate(candidates_list):
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
    
    # Compute provenance hash
    req_canonical = json.dumps(request_data, sort_keys=True, separators=(',', ':')).encode('utf-8')
    req_hash = hashlib.sha256(req_canonical).hexdigest()
    
    # Write experiment_state.json
    state_data = {
        "schema_version": "1",
        "experiment_id": experiment_id,
        "provenance": {
            "hash": req_hash,
            "algorithm": "sha256",
            "timestamp": now.isoformat(),
        },
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
    state["runs"][queued_idx]["input_hash"] = result.get("input_hash")
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

        from server.utils.artifacts import read_artifact_text
        
        runs_root = _runs_root() / "sessions"
        run_record = runs_map.get(run_id)
        input_hash = run_record.get("input_hash") if run_record else None
        
        if input_hash:
             found_run_dir = runs_root / input_hash / run_id
        else:
             # Look for run_id in sessions glob (legacy fallback)
             found_run_dir = None
             if runs_root.exists():
                 for path in runs_root.glob(f"*/{run_id}"):
                     if path.is_dir():
                         found_run_dir = path
                         break
             if not found_run_dir:
                 found_run_dir = _runs_root() / run_id
             
        # The artifact is likely in the "bundle" subdirectory for sessions
        # Try finding it in bundle first, then root of run
        if (found_run_dir / "bundle" / artifact).exists():
             return read_artifact_text(found_run_dir / "bundle", artifact, max_bytes)
             
        return read_artifact_text(found_run_dir, artifact, max_bytes)

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

@router.get("/{experiment_id}/compare-results")
def compare_experiment_results(experiment_id: str):
    """
    Get semantic comparison of two runs in an experiment (v1).
    Pure projection of results into verdicts.
    """
    # 1. Load Experiment
    exp_data = _load_experiment(experiment_id)
    if not exp_data:
        raise HTTPException(status_code=404, detail="Experiment not found")
        
    runs = exp_data.get("runs", [])
    
    # 2. Identify Runs
    # Logic: We need exactly 2 runs to compare A vs B.
    # If < 2, return comparable=False (handled by logic, but we need to feed Nones)
    
    res_a = None
    res_b = None
    
    if len(runs) >= 1:
        rid_a = runs[0].get("run_id")
        if rid_a:
            try:
                res_a = compute_result_v1(rid_a)
            except Exception:
                pass # Result loading failed (maybe run deleted or ancient)
                
    if len(runs) >= 2:
        rid_b = runs[1].get("run_id")
        if rid_b:
             try:
                 res_b = compute_result_v1(rid_b)
             except Exception:
                 pass

    # 3. Compute Comparison
    return compute_comparison_v1(experiment_id, res_a, res_b)
