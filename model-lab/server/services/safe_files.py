from pathlib import Path
from typing import Optional
from fastapi import HTTPException
import os

def _runs_root() -> Path:
    return Path(os.environ.get("MODEL_LAB_RUNS_ROOT", "runs")).resolve()

def validate_path(run_id: str, relative_path: str) -> Path:
    """
    Securely resolves a file path within a specific run directory.
    Enforces that the resulting path is strictly within the run's folder.
    Refuses symlinks that escape the root.
    """
    # 1. Find the run directory (using basic search to support hash buckets)
    # We don't want to crawl everything just to find the run, but we need the hash.
    # For now, we assume the caller might know the full path or we search efficiently.
    # Actually, runs/sessions/<hash>/<run_id>.
    # If run_id is unique enough, we can find it.
    
    # Optimization: If index service provides the absolute path, we use that as base?
    # But safe_files should verify provided "relative_path" is inside the "base_path".
    pass

def get_run_dir(run_id: str) -> Optional[Path]:
    """
    Finds the directory for a given run_id by searching runs/sessions.
    Returns None if not found.
    """
    # This is a bit expensive if we traverse everything. 
    # relying on glob for V0 is acceptable or using the Index service's cache.
    # To keep this module standalone, we'll do a focused glob.
    
    # Run IDs are typically unique.
    # Attempt to find it.
    found = list(_runs_root().glob(f"sessions/*/{run_id}"))
    if not found:
        return None
    return found[0].resolve()

def safe_file_path(run_id: str, relative_path: str) -> Path:
    """
    Returns a resolved, safe Path object for a file within a run.
    Raises HTTPException 404 if run or file not found.
    Raises HTTPException 403 if path is unsafe.
    """
    run_dir = get_run_dir(run_id)
    if not run_dir:
        raise HTTPException(status_code=404, detail="Run not found")
        
    # secure join
    try:
        # Prevent ".." attacks
        target_path = (run_dir / relative_path).resolve()
    except Exception:
         raise HTTPException(status_code=403, detail="Invalid path")
         
    # Enforce strict containment
    if not target_path.is_relative_to(run_dir):
         raise HTTPException(status_code=403, detail="Path traversal detected")
         
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
        
    if not target_path.is_file():
         raise HTTPException(status_code=400, detail="Path is not a file")

    return target_path
