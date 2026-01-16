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
