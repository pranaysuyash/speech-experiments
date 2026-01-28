from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from typing import List, Dict, Any, Optional
import hashlib
import json
import zipfile
import os
from pathlib import Path
import re

from server.services.runs_index import get_index
from server.services.safe_files import safe_file_path
from server.services.results_v1 import compute_result_v1

router = APIRouter(prefix="/api/runs", tags=["runs"])

@router.get("/{run_id}/results")
def get_run_results(run_id: str):
    """
    Get the semantic results (v1) for a run.
    Pure projection of artifacts into metrics and flags.
    """
    # Check existence via index
    if not get_index().get_run(run_id):
         raise HTTPException(status_code=404, detail="Run not found")
    
    try:
        result = compute_result_v1(run_id)
        if not result:
             # Should match index check, but verifying
             raise HTTPException(status_code=404, detail="Run not found or unreadable")
        return result
    except Exception as e:
        import logging
        logging.getLogger("server.api").error(f"Error computing results for {run_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute results")

@router.get("")
def list_runs(refresh: bool = False):
    """List all available runs from the in-memory index."""
    return get_index().list_runs(force_refresh=refresh)

@router.post("/refresh")
def refresh_runs():
    """Force a refresh of the runs index."""
    return get_index().refresh()


@router.get("/by-input/{input_hash}")
def get_runs_by_input(input_hash: str):
    """Returns all runs for a given input file hash, sorted by created_at desc."""
    runs = get_index().list_runs()
    matching = [r for r in runs if r.get("input_hash") == input_hash]
    return sorted(matching, key=lambda r: r.get("created_at", ""), reverse=True)


@router.get("/compare")
def compare_runs(run_a: str = Query(...), run_b: str = Query(...)):
    """Compare two runs side by side."""
    index = get_index()
    
    run_a_data = index.get_run(run_a)
    run_b_data = index.get_run(run_b)
    
    if not run_a_data:
        raise HTTPException(status_code=404, detail=f"Run A not found: {run_a}")
    if not run_b_data:
        raise HTTPException(status_code=404, detail=f"Run B not found: {run_b}")
    
    # Load full manifests for config comparison
    manifest_a = {}
    manifest_b = {}
    
    try:
        manifest_a = json.loads(Path(run_a_data["manifest_path"]).read_text())
    except Exception:
        pass
    
    try:
        manifest_b = json.loads(Path(run_b_data["manifest_path"]).read_text())
    except Exception:
        pass
    
    # Compute metrics from results
    metrics_a = _get_run_metrics(run_a)
    metrics_b = _get_run_metrics(run_b)
    
    # Build metrics comparison
    metrics_comparison = {}
    all_metric_keys = set(metrics_a.keys()) | set(metrics_b.keys())
    for key in all_metric_keys:
        val_a = metrics_a.get(key)
        val_b = metrics_b.get(key)
        diff = None
        if val_a is not None and val_b is not None:
            try:
                diff = val_b - val_a
            except (TypeError, ValueError):
                diff = None
        metrics_comparison[key] = {"a": val_a, "b": val_b, "diff": diff}
    
    # Config diff
    steps_a = run_a_data.get("steps_completed", [])
    steps_b = run_b_data.get("steps_completed", [])
    preprocessing_a = run_a_data.get("preprocessing_ops", [])
    preprocessing_b = run_b_data.get("preprocessing_ops", [])
    
    return {
        "runs": {
            "a": {
                "run_id": run_a,
                "status": run_a_data.get("status"),
                "started_at": run_a_data.get("started_at"),
                "input_filename": run_a_data.get("input_filename"),
                "config": manifest_a.get("config", {}),
            },
            "b": {
                "run_id": run_b,
                "status": run_b_data.get("status"),
                "started_at": run_b_data.get("started_at"),
                "input_filename": run_b_data.get("input_filename"),
                "config": manifest_b.get("config", {}),
            }
        },
        "config_diff": {
            "steps": {"a": steps_a, "b": steps_b},
            "preprocessing": {"a": preprocessing_a, "b": preprocessing_b},
        },
        "metrics_comparison": metrics_comparison,
    }


def _get_run_metrics(run_id: str) -> Dict[str, Any]:
    """Extract key metrics from a run for comparison."""
    try:
        result = compute_result_v1(run_id)
        if result and "metrics" in result:
            return {
                "transcript_word_count": result["metrics"].get("word_count"),
                "segment_count": result["metrics"].get("segment_count"),
                "duration_s": result["metrics"].get("duration_s"),
                "audio_duration_s": result["metrics"].get("audio_duration_s"),
            }
    except Exception:
        pass
    return {}


@router.post("/{run_id}/rerun")
def rerun_pipeline(run_id: str, config_overrides: Optional[Dict[str, Any]] = None):
    """Re-run a pipeline with optional config changes."""
    from server.services.lifecycle import try_acquire_worker, release_worker, launch_run_worker, RunnerBusyError
    from harness.session import SessionRunner
    
    index = get_index()
    run_data = index.get_run(run_id)
    
    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")
    
    # Load run_request.json to get original input file
    run_request_path = Path(run_data["root_path"]) / "run_request.json"
    if not run_request_path.exists():
        raise HTTPException(status_code=400, detail="Cannot rerun: run_request.json missing")
    
    try:
        run_request = json.loads(run_request_path.read_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read run request: {e}")
    
    # Find original input file - check manifest for input_path
    manifest_path = Path(run_data["manifest_path"])
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read manifest")
    
    input_path_str = manifest.get("input_path")
    if not input_path_str:
        raise HTTPException(status_code=400, detail="Cannot rerun: original input path not found")
    
    input_path = Path(input_path_str)
    if not input_path.exists():
        raise HTTPException(status_code=400, detail=f"Cannot rerun: input file missing: {input_path}")
    
    # Get original configuration
    original_steps = run_request.get("steps_requested")
    original_config = run_request.get("config", {})
    preprocessing = run_request.get("preprocessing", [])
    
    # Merge config overrides
    merged_config = {**original_config, **(config_overrides or {})}
    
    # Acquire worker
    if not try_acquire_worker():
        raise HTTPException(status_code=409, detail="Runner is busy with another job")
    
    try:
        runs_root = Path(os.environ.get("MODEL_LAB_RUNS_ROOT", "runs")).resolve()
        runner = SessionRunner(
            input_path,
            runs_root,
            steps=original_steps,
            config=merged_config,
        )
        
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        
        # Create new run_request with parent reference
        new_run_request = {
            "schema_version": "1",
            "requested_at": now.isoformat(),
            "source": "rerun",
            "parent_run_id": run_id,
            "use_case_id": run_request.get("use_case_id", "rerun"),
            "steps_preset": run_request.get("steps_preset", "custom"),
            "steps_requested": original_steps,
            "filename_original": input_path.name,
            "sha256": run_request.get("sha256"),
            "config": merged_config,
            "preprocessing": preprocessing,
        }
        
        result = launch_run_worker(runner, new_run_request, background=True)
        
        return {
            "run_id": runner.run_id,
            "parent_run_id": run_id,
            "console_url": f"/runs/{runner.run_id}",
            "worker_pid": result.get("worker_pid"),
        }
    except Exception:
        release_worker()
        raise

# Canonical pipeline step order for progress display
PIPELINE_STEP_ORDER = [
    "ingest",
    "asr",
    "diarization",
    "alignment",
    "chapters",
    "summarize_by_speaker",
    "action_items_assignee",
    "bundle",
]


def _build_steps_progress(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build ordered steps_progress array from manifest.
    
    Returns list of step progress dicts with:
    - name: step key
    - status: PENDING | RUNNING | COMPLETED | FAILED | SKIPPED
    - progress_pct: 0-100
    - message: optional progress message
    - duration_ms: for completed steps
    - estimated_remaining_s: for running steps
    """
    manifest_steps = manifest.get("steps", {})
    steps_requested = manifest.get("steps_requested", [])
    
    # Use steps_requested if available, otherwise use canonical order
    if steps_requested:
        ordered_steps = steps_requested
    else:
        # Use manifest steps keys in canonical order, falling back to alphabetical
        ordered_steps = [s for s in PIPELINE_STEP_ORDER if s in manifest_steps]
        # Add any remaining steps not in canonical order
        for s in sorted(manifest_steps.keys()):
            if s not in ordered_steps:
                ordered_steps.append(s)
    
    result = []
    for step_name in ordered_steps:
        step_data = manifest_steps.get(step_name, {})
        step_status = step_data.get("status", "PENDING")
        
        progress_entry: Dict[str, Any] = {
            "name": step_name,
            "status": step_status,
            "progress_pct": step_data.get("progress_pct", 0 if step_status in ("PENDING", "RUNNING") else 100 if step_status in ("COMPLETED", "SKIPPED") else 0),
        }
        
        # Add optional fields
        if step_data.get("progress_message"):
            progress_entry["message"] = step_data["progress_message"]
        
        if step_data.get("duration_ms") is not None:
            progress_entry["duration_ms"] = step_data["duration_ms"]
        
        if step_data.get("estimated_remaining_s") is not None:
            progress_entry["estimated_remaining_s"] = step_data["estimated_remaining_s"]
        
        if step_data.get("started_at"):
            progress_entry["started_at"] = step_data["started_at"]
        
        if step_data.get("ended_at"):
            progress_entry["ended_at"] = step_data["ended_at"]
        
        result.append(progress_entry)
    
    return result


def _derive_status_config(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Return a status-friendly config payload populated from manifest + step metadata."""
    config_source = manifest.get("config") or {}
    config: Dict[str, Any] = dict(config_source)
    steps = manifest.get("steps", {}) or {}

    asr_step = steps.get("asr", {}) or {}
    resolved_asr = asr_step.get("resolved_config") or {}
    asr_config = dict(config.get("asr") or {})

    if resolved_asr:
        if resolved_asr.get("model_id"):
            asr_config.setdefault("model_id", resolved_asr["model_id"])
        if resolved_asr.get("model_name"):
            asr_config.setdefault("model_name", resolved_asr["model_name"])
        elif resolved_asr.get("model_id"):
            asr_config.setdefault("model_name", resolved_asr["model_id"])

        for field in ("source", "device", "language"):
            if resolved_asr.get(field):
                asr_config[field] = resolved_asr[field]

    config["asr"] = asr_config

    diarization_config = dict(config.get("diarization") or {})
    if diarization_config.get("enabled") is None:
        diarization_config["enabled"] = "diarization" in steps
    config["diarization"] = diarization_config

    return config


@router.get("/{run_id}/status")
def get_run_status(run_id: str):
    """Get lightweight status for a run with stale detection."""
    from datetime import datetime, timezone
    
    run = get_index().get_run(run_id)
    if not run:
         raise HTTPException(status_code=404, detail="Run not found")
    
    # --- STALE CONTRACT HARDENING (Phase 2) ---
    # Ensure consistence payload regardless of live/stale status.
    # Source Logic:
    # 1. Manifest read success -> "manifest"
    # 2. Manifest read fail -> "index" (with snapshot_reason="manifest_missing")
    
    snapshot_source = "manifest"
    snapshot_reason = None
    manifest_mtime = 0
    
    try:
        manifest_path = Path(run["manifest_path"])
        manifest = json.loads(manifest_path.read_text())
        manifest_mtime = manifest_path.stat().st_mtime
    except FileNotFoundError:
        snapshot_source = "index"
        snapshot_reason = "manifest_missing"
        # Fallback to index data (better than 500)
        # But explicitly mark as degraded
        manifest = {
            "status": run["status"],
            "steps": {k: {"status": "COMPLETED"} for k in run.get("steps_completed", [])},
            "error": {"type": "SnapshotError", "message": "Run manifest missing on disk"},
        }
    except json.JSONDecodeError:
        snapshot_source = "index"
        snapshot_reason = "manifest_corrupt"
         # Fallback to index data
        manifest = {
             "status": run["status"],
             "steps": {k: {"status": "COMPLETED"} for k in run.get("steps_completed", [])},
             "error": {"type": "SnapshotError", "message": "Run manifest corrupt"},
        }

    status = manifest.get("status", run["status"])
    current_step = manifest.get("current_step")
    updated_at = manifest.get("updated_at")

    # Stale detection based on updated_at
    STALE_THRESHOLD_SECONDS = 90
    if status == "RUNNING" and updated_at:
        try:
            last_update = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            elapsed = (now - last_update).total_seconds()
            
            if elapsed > STALE_THRESHOLD_SECONDS:
                status = "STALE"
                snapshot_reason = f"no_heartbeat_{int(elapsed)}s"
                if "error" not in manifest:
                     manifest["error"] = {
                         "type": "StaleRun",
                         "message": f"No heartbeat in {int(elapsed)}s"
                     }
        except Exception:
            pass

    # Authoritative Lists
    # Use manifest steps keys if available (most accurate)
    steps_completed = []
    if "steps" in manifest:
        # Filter for COMPLETED steps
        steps_completed = [
            k for k, v in manifest["steps"].items() 
            if v.get("status") in ("COMPLETED", "SKIPPED")
        ]
    else:
        # Fallback to index
        steps_completed = run.get("steps_completed", [])

    # Build steps_progress array for real-time pipeline visibility
    steps_progress = _build_steps_progress(manifest)

    status_config = _derive_status_config(manifest)

    return {
        "run_id": run_id,
        "status": status,
        "current_step": current_step,
        "updated_at": updated_at,
        "steps_completed": steps_completed,
        "steps_progress": steps_progress,
        "failure_step": manifest.get("failure_step"),
        "error_message": manifest.get("error", {}).get("message"),
        "input_metadata": manifest.get("input_metadata", {}),
        "input_hash": run.get("input_hash"),  # For run history linking
        "config": status_config,
        "artifacts": manifest.get("artifacts_by_type", {}), # Use new global index
        "resolved_device": run.get("config", {}).get("resolved_device"), # From index
        "meta": {
            "snapshot_source": snapshot_source,
            "snapshot_reason": snapshot_reason,
            "manifest_mtime": manifest_mtime
        }
    }

@router.get("/{run_id}/details")
def get_run_details_v2(run_id: str):
    """Get full manifest for a specific run, with enhanced details."""
    from datetime import datetime, timezone

    run = get_index().get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found in index (try refresh?)")

    manifest_path = Path(run["manifest_path"])
    manifest = {}
    try:
        manifest = json.loads(manifest_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        # If manifest is missing or corrupt, we still want to return some info
        # Fallback to index data for basic status
        manifest = {
            "status": run["status"],
            "steps": {k: {"status": "COMPLETED"} for k in run.get("steps_completed", [])},
            "error": {"type": "SnapshotError", "message": "Run manifest missing or corrupt"},
        }

    status = manifest.get("status", run["status"])
    current_step = manifest.get("current_step")
    updated_at = manifest.get("updated_at")
    last_semantic_progress_at = manifest.get("last_semantic_progress_at")

    # Stale detection based on updated_at
    STALE_THRESHOLD_SECONDS = 90
    is_stalled = False
    if status == "RUNNING" and updated_at:
        try:
            last_update = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            elapsed = (now - last_update).total_seconds()
            
            if elapsed > STALE_THRESHOLD_SECONDS:
                status = "STALE"
                is_stalled = True
        except Exception:
            pass # Ignore parsing errors for updated_at

    # Input metadata
    input_meta = manifest.get("input_metadata", {})
    if not input_meta and run.get("input_metadata"):
        input_meta = run["input_metadata"] # Fallback to index if manifest is empty

    # Artifacts availability (global index)
    artifacts = manifest.get("artifacts_by_type", {})

    # Extract per-step details with errors
    steps = []
    if "steps" in manifest:
        for step_name, step_data in manifest["steps"].items():
            step_info = {
                "name": step_name,
                "status": step_data.get("status", "PENDING"),
                "started_at": step_data.get("started_at"),
                "ended_at": step_data.get("ended_at"),
                "duration_ms": step_data.get("duration_ms"),
                "resolved_config": step_data.get("resolved_config"),
            }
            
            # Add artifacts (filter out internal fields like path, content_type)
            raw_artifacts = step_data.get("artifacts", [])
            api_artifacts = []
            for art in raw_artifacts:
                if "id" in art:
                    # New semantic schema - expose safe fields only
                    api_artifacts.append({
                        "id": art.get("id"),
                        "filename": art.get("filename"),
                        "role": art.get("role"),
                        "produced_by": art.get("produced_by"),
                        "size_bytes": art.get("size_bytes"),
                        "downloadable": art.get("downloadable", False),
                    })
                # Legacy format artifacts are not exposed to prevent confusion
            
            if api_artifacts:
                step_info["artifacts"] = api_artifacts
            
            # Add error details if step failed
            if step_info["status"] == "FAILED":
                error_info = {}
                if "error" in step_data:
                    error_info["type"] = step_data["error"].get("type", "UnknownError")
                    error_info["message"] = step_data["error"].get("message", "No error message available")
                else:
                    # Fallback if error field missing
                    error_info["type"] = "UnknownError"
                    error_info["message"] = "Step failed without error details"
                step_info["error"] = error_info
            
            steps.append(step_info)



    return {
        "run_id": run_id,
        "status": status,
        "started_at": manifest.get("started_at"),
        "steps_completed": run.get("steps_completed", []),
        "current_step": current_step,
        "updated_at": updated_at,
        "last_semantic_progress_at": last_semantic_progress_at,
        "is_stalled": is_stalled,
        "failure_step": manifest.get("failure_step"),
        "input_metadata": input_meta,
        "config": manifest.get("config", {}),
        "artifacts_availability": artifacts,
        "steps": steps
    }

@router.get("/{run_id}")
def get_run_details(run_id: str):
    """Get full manifest for a specific run."""
    run = get_index().get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found in index (try refresh?)")
    
    # We could return the already indexed summary, but usually detail view wants the full manifest.
    # The index stores 'manifest_path'.
    manifest_path = safe_file_path(run_id, "manifest.json")
    try:
        data = json.loads(manifest_path.read_text())
        # Enhance with resolved paths for UI convenience?
        # For now, just return raw manifest + summary info
        return {
            "summary": run,
            "manifest": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read manifest: {str(e)}")

@router.get("/{run_id}/transcript")
def get_run_transcript(run_id: str):
    """
    Returns a Normalized Transcript DTO.
    DTO: { "segments": [...], "chapters": [...] }
    """
    try:
        transcript = get_index().get_transcript(run_id)
        if not transcript:
             if get_index().get_run(run_id):
                 return { "run_id": run_id, "segments": [], "chapters": [] }
             raise HTTPException(status_code=404, detail="Run not found")
        return transcript
    except RuntimeError as e:
        if "E_ARTIFACT_REGISTRY_MISSING" in str(e):
             raise HTTPException(status_code=400, detail=str(e))
        raise e

@router.get("/{run_id}/search")
def search_run(run_id: str, q: str = Query(..., min_length=2), limit: int = 50):
    """
    Search within a run.
    """
    run = get_index().get_run(run_id)
    if not run:
         raise HTTPException(status_code=404, detail="Run not found")
    
    return get_index().search_run(run_id, q, limit)

@router.get("/{run_id}/audio")
def stream_audio(run_id: str):
    """Stream the processed audio file."""
    # Standard path: artifacts/ingest/processed_audio.wav
    # Or read from manifest['steps']['ingest']['result']['processed_audio_path']?
    # That path might be absolute.
    # We prefer relative to run dir if possible.
    # But Ingest stores absolute paths typically.
    
    # Secure approach: Look for 'artifacts/ingest/*.wav' or use specific name.
    # Current pipeline: 'artifacts/ingest/processed_audio.wav'
    
    try:
        path = safe_file_path(run_id, "artifacts/ingest/processed_audio.wav")
        return FileResponse(path, media_type="audio/wav")
    except HTTPException:
        # Fallback to finding ANY wav in ingest?
        raise HTTPException(status_code=404, detail="Audio not found")

@router.get("/{run_id}/bundle")
def get_bundle_manifest(run_id: str):
    """Return Meeting Pack bundle_manifest.json."""
    path = safe_file_path(run_id, "bundle/bundle_manifest.json")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read bundle manifest: {e}")


_ARTIFACT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _validate_artifact_name(name: str) -> None:
    # FastAPI will percent-decode route params, but we hard-reject anything that could
    # be interpreted differently by downstream components (double decode, windows paths, etc).
    if not name or not isinstance(name, str):
        raise HTTPException(status_code=400, detail="Invalid artifact name")
    if "\x00" in name or "%" in name or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid artifact name")
    if not _ARTIFACT_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="Invalid artifact name")


def _load_bundle_manifest(run_id: str) -> Dict[str, Any]:
    p = safe_file_path(run_id, "bundle/bundle_manifest.json")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read bundle manifest: {e}")


def _artifact_rel_path(entry: Dict[str, Any]) -> Optional[str]:
    # v0.1 uses rel_path; older versions may use path.
    rp = entry.get("rel_path")
    if isinstance(rp, str) and rp:
        return rp
    rp = entry.get("path")
    if isinstance(rp, str) and rp:
        return rp
    return None


def _resolve_manifest_listed_artifact(run_id: str, artifact_name: str) -> Dict[str, Any]:
    """
    Enforce allowlist-by-manifest: only artifacts listed in bundle_manifest.json can be served.
    Returns {rel_path, content_type}.
    """
    if artifact_name == "bundle_manifest.json":
        return {"rel_path": "bundle/bundle_manifest.json", "content_type": "application/json"}

    manifest = _load_bundle_manifest(run_id)
    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise HTTPException(status_code=500, detail="Invalid bundle manifest")

    match = None
    for a in artifacts:
        if isinstance(a, dict) and a.get("name") == artifact_name:
            match = a
            break
    if not match:
        raise HTTPException(status_code=404, detail="Artifact not found")

    rel_path = _artifact_rel_path(match)
    if not rel_path:
        raise HTTPException(status_code=500, detail="Invalid bundle manifest artifact entry")

    # Must be exactly under bundle/ and match the requested name.
    rel_path_norm = rel_path.replace("\\", "/")
    if rel_path_norm != f"bundle/{artifact_name}":
        raise HTTPException(status_code=403, detail="Invalid artifact path")

    content_type = match.get("content_type")
    if not isinstance(content_type, str) or not content_type:
        # Conservative fallback
        content_type = "application/octet-stream"

    return {"rel_path": rel_path_norm, "content_type": content_type}


@router.get("/{run_id}/bundle/{artifact_name}")
def download_bundle_artifact(run_id: str, artifact_name: str, max_bytes: Optional[int] = Query(default=None, ge=1, le=2_000_000)):
    """Stream a single Meeting Pack artifact from bundle/ safely.

    If max_bytes is provided, returns at most that many bytes (for UI previews).
    """
    _validate_artifact_name(artifact_name)
    resolved = _resolve_manifest_listed_artifact(run_id, artifact_name)
    content_type = resolved["content_type"]
    path = safe_file_path(run_id, resolved["rel_path"])

    if max_bytes is not None:
        # Preview mode: cap content, never stream huge files into the browser.
        # Only allow for text-ish formats.
        if not (content_type.startswith("text/") or content_type == "application/json"):
            raise HTTPException(status_code=400, detail="Preview not supported for this artifact")
        if path.stat().st_size > max_bytes:
            raise HTTPException(
                status_code=413,
                detail={"error_code": "PREVIEW_TOO_LARGE", "message": "Preview too large; download instead."},
            )
        with open(path, "rb") as f:
            data = f.read(max_bytes)
        # Best-effort UTF-8 decode for preview.
        text = data.decode("utf-8", errors="replace")
        return Response(content=text, media_type=content_type)

    return FileResponse(path, media_type=content_type, filename=artifact_name)


@router.get("/{run_id}/bundle.zip")
def download_bundle_zip(run_id: str):
    """Zip Meeting Pack artifacts on-demand and stream the zip."""
    # Ensure manifest exists
    run_dir = safe_file_path(run_id, "manifest.json").parent
    manifest_path = safe_file_path(run_id, "bundle/bundle_manifest.json")
    bundle_manifest = _load_bundle_manifest(run_id)
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    bundle_dir = run_dir / "bundle"
    zip_path = bundle_dir / f"meeting_pack_{manifest_sha256[:16]}.zip"

    files: List[Dict[str, str]] = []
    files.append({"name": "bundle_manifest.json", "path": str(manifest_path)})
    for a in bundle_manifest.get("artifacts", []):
        if not isinstance(a, dict):
            continue
        name = a.get("name")
        if not isinstance(name, str):
            continue
        _validate_artifact_name(name)
        rel_path = _artifact_rel_path(a)
        if not rel_path:
            continue
        rel_path_norm = rel_path.replace("\\", "/")
        if rel_path_norm != f"bundle/{name}":
            continue
        try:
            p = safe_file_path(run_id, rel_path_norm)
        except HTTPException:
            continue
        files.append({"name": name, "path": str(p)})

    if not files:
        raise HTTPException(status_code=404, detail="No bundle artifacts found")

    # Cache: key by sha256(bundle_manifest.json) so repeated calls are cheap and stable.
    if zip_path.exists():
        return FileResponse(zip_path, media_type="application/zip", filename="meeting_pack.zip")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_zip = zip_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(files, key=lambda x: x["name"]):
            # arcname is the bare filename to avoid zip-slip/path surprises
            arcname = f["name"]
            _validate_artifact_name(arcname)
            zf.write(f["path"], arcname=arcname)
    os.replace(tmp_zip, zip_path)

    return FileResponse(zip_path, media_type="application/zip", filename="meeting_pack.zip")


@router.get("/{run_id}/session_bundle.zip")
def download_session_bundle_zip(run_id: str):
    """Download the legacy session_bundle.zip if present."""
    path = safe_file_path(run_id, "bundle/session_bundle.zip")
    return FileResponse(path, media_type="application/zip", filename="session_bundle.zip")


@router.get("/{run_id}/artifacts/{artifact_id}")
def download_artifact(run_id: str, artifact_id: str):
    """
    Download an artifact by ID.
    
    Phase 2.5 invariants (non-negotiable):
    1. Resolve ONLY via manifest - no filesystem probing
    2. Path traversal impossible - artifact_id is opaque lookup key
    3. 404 if artifact not declared in manifest
    4. 403 if downloadable=false
    5. Stream via FileResponse with correct Content-Type
    """
    import logging
    logger = logging.getLogger("server.api")
    
    # 1. Get run from index
    run = get_index().get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    # 2. Read manifest to find artifact
    manifest_path = Path(run["manifest_path"])
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Manifest not found")
    
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as e:
        logger.error(f"Failed to read manifest for {run_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to read manifest")
    
    # 3. Search for artifact by ID across all steps
    session_dir = manifest_path.parent
    found_artifact = None
    
    for step_name, step_data in manifest.get("steps", {}).items():
        for art in step_data.get("artifacts", []):
            # Only new schema artifacts have 'id'
            if art.get("id") == artifact_id:
                found_artifact = art
                break
        if found_artifact:
            break
    
    if not found_artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    # 4. Check downloadable flag (403 if false)
    if not found_artifact.get("downloadable", False):
        raise HTTPException(status_code=403, detail="Artifact not downloadable")
    
    # 5. Resolve file path (relative to session_dir, from manifest)
    relative_path = found_artifact.get("path")
    if not relative_path:
        raise HTTPException(status_code=500, detail="Artifact path missing")
    
    # Security: ensure resolved path stays within session_dir (prevents ../, absolute paths, and symlink escape).
    session_dir_resolved = session_dir.resolve()
    file_path = (session_dir / relative_path).resolve()
    try:
        file_path.relative_to(session_dir_resolved)
    except ValueError:
        logger.warning(f"Path traversal attempt blocked: {relative_path}")
        raise HTTPException(status_code=403, detail="Invalid artifact path")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file not found on disk")
    
    # 6. Return file with proper Content-Type
    content_type = found_artifact.get("content_type", "application/octet-stream")
    filename = found_artifact.get("filename", file_path.name)
    
    return FileResponse(
        file_path,
        media_type=content_type,
        filename=filename
    )
