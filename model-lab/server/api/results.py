"""
Results and Findings API endpoints.

Provides read-only access to eval.json files and aggregated findings.
"""
from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

from server.services.runs_index import get_index
from server.api.eval_loader import load_eval

router = APIRouter(prefix="/api", tags=["results"])


@router.get("/runs/{run_id}/eval")
def get_run_eval(run_id: str):
    """
    Get evaluation results for a specific run.
    Returns 404 if eval.json doesn't exist.
    """
    run = get_index().get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    manifest_path = Path(run["manifest_path"])
    run_root = manifest_path.parent
    
    eval_data = load_eval(run_root)
    if eval_data:
        return eval_data
    
    raise HTTPException(status_code=404, detail="Evaluation not available for this run")


@router.get("/results")
def get_results(
    use_case_id: Optional[str] = Query(None),
    model_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None)
):
    """
    Get list of runs enriched with eval summary if present.
    Supports filtering by use_case_id, model_id, and status.
    """
    runs = get_index().list_runs()
    
    results = []
    for run in runs:
        # Apply filters
        if status and run["status"] != status:
            continue
        
        # Build result entry
        result = {
            "run_id": run["run_id"],
            "status": run["status"],
            "started_at": run.get("started_at"),
            "duration_ms": run.get("duration_ms"),
            "steps_completed": run.get("steps_completed", []),
            "eval_available": False,
            "use_case_id": None,
            "model_id": None,
            "metrics": {},
            "checks_passed": None,
            "checks_total": None
        }
        
        # Try to load eval if it exists
        manifest_path = Path(run["manifest_path"])
        run_root = manifest_path.parent
        eval_data = load_eval(run_root)
        
        if eval_data:
            result["eval_available"] = True
            result["use_case_id"] = eval_data.get("use_case_id")
            result["model_id"] = eval_data.get("model_id")
            result["metrics"] = eval_data.get("metrics", {})
            result["score_cards"] = eval_data.get("score_cards", [])
            
            checks = eval_data.get("checks", [])
            result["checks_total"] = len(checks)
            result["checks_passed"] = sum(1 for c in checks if c.get("passed"))
        
        # Apply eval-based filters
        if use_case_id and result["use_case_id"] != use_case_id:
            continue
        if model_id and result["model_id"] != model_id:
            continue
        
        results.append(result)
    
    # Sort by started_at descending (newest first)
    results.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    
    return results


@router.get("/findings")
def get_findings(
    severity: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    model_id: Optional[str] = Query(None),
    use_case_id: Optional[str] = Query(None)
):
    """
    Aggregate findings across runs by finding_id.
    Returns: finding_id, title, category, severity, count, first_seen_at, last_seen_at, latest_run_id.
    """
    runs = get_index().list_runs()
    
    # Collect all findings across runs
    findings_map: Dict[str, Dict[str, Any]] = {}
    
    for run in runs:
        manifest_path = Path(run["manifest_path"])
        run_root = manifest_path.parent
        eval_data = load_eval(run_root)
        
        if not eval_data:
            continue
        
        # Apply filters at run level
        if use_case_id and eval_data.get("use_case_id") != use_case_id:
            continue
        if model_id and eval_data.get("model_id") != model_id:
            continue
        
        # Process findings
        for finding in eval_data.get("findings", []):
            fid = finding.get("finding_id")
            if not fid:
                # Generate stable ID from title + category for V1
                fid = f"{finding.get('category', 'unknown')}:{finding.get('title', 'untitled')}"
            
            # Apply finding-level filters
            if severity and finding.get("severity") != severity:
                continue
            if category and finding.get("category") != category:
                continue
            
            # Aggregate
            if fid not in findings_map:
                findings_map[fid] = {
                    "finding_id": fid,
                    "title": finding.get("title", "Untitled"),
                    "category": finding.get("category", "unknown"),
                    "severity": finding.get("severity", "low"),
                    "details": finding.get("details", ""),
                    "count": 0,
                    "first_seen_at": run.get("started_at"),
                    "last_seen_at": run.get("started_at"),
                    "latest_run_id": run["run_id"],
                    "evidence_paths": finding.get("evidence_paths", [])
                }
            
            entry = findings_map[fid]
            entry["count"] += 1
            
            # Update timestamps
            run_time = run.get("started_at", "")
            if run_time > entry["last_seen_at"]:
                entry["last_seen_at"] = run_time
                entry["latest_run_id"] = run["run_id"]
                entry["evidence_paths"] = finding.get("evidence_paths", [])
            if run_time < entry["first_seen_at"]:
                entry["first_seen_at"] = run_time
    
    # Convert to list and sort by last_seen_at descending
    findings_list = list(findings_map.values())
    findings_list.sort(key=lambda f: f["last_seen_at"], reverse=True)
    
    return findings_list
