from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from typing import List, Dict, Any, Optional
import json
import zipfile

from server.services.runs_index import get_index
from server.services.safe_files import safe_file_path

router = APIRouter(prefix="/api/runs", tags=["runs"])

@router.get("")
def list_runs(refresh: bool = False):
    """List all available runs from the in-memory index."""
    return get_index().list_runs(force_refresh=refresh)

@router.post("/refresh")
def refresh_runs():
    """Force a refresh of the runs index."""
    return get_index().refresh()

@router.get("/{run_id}/status")
def get_run_status(run_id: str):
    """Get lightweight status for a run with stale detection."""
    from datetime import datetime, timezone
    
    run = get_index().get_run(run_id)
    if not run:
         raise HTTPException(status_code=404, detail="Run not found")
    
    # Get full manifest for updated_at and current_step
    from pathlib import Path
    import json
    manifest_path = Path(run["manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    
    status = run["status"]
    current_step = manifest.get("current_step")
    updated_at = manifest.get("updated_at")
    
    # Stale detection: if RUNNING but no heartbeat for > 90s, mark as STALE
    STALE_THRESHOLD_SECONDS = 90
    
    if status == "RUNNING" and updated_at:
        try:
            # Parse ISO timestamp
            last_update = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            elapsed = (now - last_update).total_seconds()
            
            if elapsed > STALE_THRESHOLD_SECONDS:
                status = "STALE"
                # Add error fields for frontend display
                return {
                    "run_id": run_id,
                    "status": status,
                    "started_at": run.get("started_at"),
                    "steps_completed": run.get("steps_completed", []),
                    "current_step": current_step,
                    "updated_at": updated_at,
                    "error_code": "STALE_RUN",
                    "error_message": f"No heartbeat in {int(elapsed)}s"
                }
        except (ValueError, TypeError):
            pass  # If timestamp parsing fails, just return normal status
    
    # Return minimal status from index cache
    return {
        "run_id": run_id,
        "status": status,
        "started_at": run.get("started_at"),
        "steps_completed": run.get("steps_completed", []),
        "current_step": current_step,
        "updated_at": updated_at
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
    transcript = get_index().get_transcript(run_id)
    if not transcript:
         # Could be 404 or just empty if no transcript yet
         # If run exists but no transcript, return empty structure
         run = get_index().get_run(run_id)
         if not run:
             raise HTTPException(status_code=404, detail="Run not found")
         return { "run_id": run_id, "segments": [], "chapters": [] }
    
    return transcript

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


_BUNDLE_ALLOWED_FILES = {
    "bundle_manifest.json": "application/json",
    "transcript.json": "application/json",
    "summary.md": "text/markdown",
    "action_items.csv": "text/csv",
    "decisions.md": "text/markdown",
    "chapters.json": "application/json",
    "entities.json": "application/json",
}


@router.get("/{run_id}/bundle/{artifact_name}")
def download_bundle_artifact(run_id: str, artifact_name: str):
    """Stream a single Meeting Pack artifact from bundle/ safely."""
    if "/" in artifact_name or "\\" in artifact_name:
        raise HTTPException(status_code=400, detail="Invalid artifact name")
    content_type = _BUNDLE_ALLOWED_FILES.get(artifact_name)
    if not content_type:
        raise HTTPException(status_code=404, detail="Unknown artifact")

    path = safe_file_path(run_id, f"bundle/{artifact_name}")
    return FileResponse(path, media_type=content_type, filename=artifact_name)


@router.get("/{run_id}/bundle.zip")
def download_bundle_zip(run_id: str):
    """Zip Meeting Pack artifacts on-demand and stream the zip."""
    # Ensure manifest exists
    run_dir = safe_file_path(run_id, "manifest.json").parent
    manifest_path = safe_file_path(run_id, "bundle/bundle_manifest.json")
    try:
        bundle_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read bundle manifest: {e}")

    bundle_dir = run_dir / "bundle"
    zip_path = bundle_dir / "meeting_pack.zip"

    files: List[Dict[str, str]] = []
    files.append({"name": "bundle_manifest.json", "path": str(manifest_path)})
    for a in bundle_manifest.get("artifacts", []):
        name = a.get("name")
        rel_path = a.get("path")
        if not isinstance(name, str) or not isinstance(rel_path, str):
            continue
        if name not in _BUNDLE_ALLOWED_FILES:
            continue
        try:
            p = safe_file_path(run_id, rel_path)
        except HTTPException:
            continue
        files.append({"name": name, "path": str(p)})

    if not files:
        raise HTTPException(status_code=404, detail="No bundle artifacts found")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(files, key=lambda x: x["name"]):
            zf.write(f["path"], arcname=f["name"])

    return FileResponse(zip_path, media_type="application/zip", filename="meeting_pack.zip")


@router.get("/{run_id}/session_bundle.zip")
def download_session_bundle_zip(run_id: str):
    """Download the legacy session_bundle.zip if present."""
    path = safe_file_path(run_id, "bundle/session_bundle.zip")
    return FileResponse(path, media_type="application/zip", filename="session_bundle.zip")
