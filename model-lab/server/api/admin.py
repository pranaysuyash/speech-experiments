"""
Admin API - System management endpoints.

Provides:
- Disk usage monitoring
- Run cleanup
- System health
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from server.services.runs_index import cleanup_old_runs, get_disk_usage

router = APIRouter(prefix="/api/admin", tags=["admin"])

# Default retention period in days
DEFAULT_RETENTION_DAYS = 30


def _min_free_bytes() -> int:
    """Minimum free bytes required. Configurable via env."""
    return int(
        os.environ.get("MODEL_LAB_MIN_FREE_BYTES", str(5 * 1024 * 1024 * 1024))
    )  # 5GB default


@router.get("/disk-usage")
def get_disk_usage_endpoint() -> JSONResponse:
    """
    Get disk usage for the runs directory.

    Returns total, used, and free bytes, plus run count.
    """
    usage = get_disk_usage()
    return JSONResponse(content=usage)


@router.get("/disk-usage/check")
def check_disk_space_endpoint() -> JSONResponse:
    """
    Check if there's enough disk space for new runs.

    Returns:
        - ok: true if sufficient space, false otherwise
        - free_bytes: current free space
        - min_required: minimum required space
    """
    usage = get_disk_usage()
    min_required = _min_free_bytes()

    ok = usage["free_bytes"] >= min_required

    return JSONResponse(
        content={
            "ok": ok,
            "free_bytes": usage["free_bytes"],
            "min_required": min_required,
            "message": "Sufficient disk space"
            if ok
            else f"Low disk space: {usage['free_bytes'] / (1024**3):.1f}GB free, need {min_required / (1024**3):.1f}GB",
        }
    )


@router.post("/cleanup")
def cleanup_endpoint(
    retention_days: int = Query(DEFAULT_RETENTION_DAYS, ge=1, le=365),
    dry_run: bool = Query(False),
) -> JSONResponse:
    """
    Clean up old runs.

    Args:
        retention_days: Delete runs not modified in this many days (default 30)
        dry_run: If true, only report what would be deleted without actually deleting

    Returns:
        - deleted: list of run IDs that were/would be deleted
        - freed_bytes: bytes freed
        - errors: any errors encountered
    """
    result = cleanup_old_runs(retention_days=retention_days, dry_run=dry_run)
    return JSONResponse(content=result)


@router.get("/health")
def health_check_endpoint() -> JSONResponse:
    """
    Comprehensive health check with dependency verification.

    Checks:
    - runs directory writable
    - disk space above threshold
    - ability to list runs
    """
    import logging

    logger = logging.getLogger("server.admin")
    checks = {}
    overall_status = "healthy"

    # Check 1: runs directory writable
    try:
        runs_root = Path(os.environ.get("MODEL_LAB_RUNS_ROOT", "runs"))
        test_file = runs_root / ".health_check_write"
        test_file.write_text("ok")
        test_file.unlink()
        checks["runs_dir_writable"] = {"status": "ok"}
    except Exception as e:
        checks["runs_dir_writable"] = {"status": "error", "message": str(e)}
        overall_status = "degraded"

    # Check 2: disk space
    try:
        usage = get_disk_usage()
        min_free = _min_free_bytes()
        if usage["free_bytes"] < min_free:
            checks["disk_space"] = {
                "status": "warning",
                "free_bytes": usage["free_bytes"],
                "min_required": min_free,
            }
            if overall_status == "healthy":
                overall_status = "degraded"
        else:
            checks["disk_space"] = {"status": "ok", "free_bytes": usage["free_bytes"]}
    except Exception as e:
        checks["disk_space"] = {"status": "error", "message": str(e)}
        overall_status = "unhealthy"

    # Check 3: can list runs
    try:
        from server.services.runs_index import get_index

        runs = get_index().list_runs()
        checks["run_index"] = {"status": "ok", "run_count": len(runs)}
    except Exception as e:
        checks["run_index"] = {"status": "error", "message": str(e)}
        overall_status = "unhealthy"

    logger.info(f"Health check: {overall_status}", extra={"checks": checks})

    return JSONResponse(
        content={
            "status": overall_status,
            "checks": checks,
        }
    )
