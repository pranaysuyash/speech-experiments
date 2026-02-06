"""
WebSocket endpoint for real-time run progress streaming.

Provides:
- /api/runs/{run_id}/ws: Stream events for a run
"""

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("server.ws")

router = APIRouter(tags=["websocket"])


def _runs_root() -> Path:
    """Get the runs root directory."""
    return Path("data/runs")


def _get_run_dir(run_id: str) -> Path | None:
    """
    Find the run directory for a given run_id.

    Run directories are stored as: data/runs/sessions/{input_hash}/{run_id}/
    """
    runs_root = _runs_root()
    sessions_dir = runs_root / "sessions"

    if not sessions_dir.exists():
        return None

    # Search for the run_id in session directories
    for input_hash_dir in sessions_dir.iterdir():
        if input_hash_dir.is_dir():
            run_dir = input_hash_dir / run_id
            if run_dir.exists() and run_dir.is_dir():
                return run_dir

    return None


def _get_events_file(run_dir: Path) -> Path:
    """Get the events file path for a run directory."""
    return run_dir / "events.jsonl"


@router.websocket("/api/runs/{run_id}/ws")
async def run_websocket(websocket: WebSocket, run_id: str):
    """
    WebSocket endpoint for streaming run events.

    Clients connect to receive real-time updates about run progress.
    The endpoint:
    1. Sends all existing events immediately
    2. Tails the events file for new events
    3. Sends heartbeats to keep the connection alive
    4. Closes when the run completes or client disconnects
    """
    await websocket.accept()
    logger.info(f"WebSocket connected for run {run_id}")

    run_dir = _get_run_dir(run_id)
    if not run_dir:
        await websocket.send_json({"type": "error", "message": f"Run not found: {run_id}"})
        await websocket.close(code=4004)
        return

    events_file = _get_events_file(run_dir)
    manifest_file = run_dir / "manifest.json"

    # Track file position for tailing
    last_pos = 0

    try:
        # Send existing events first
        if events_file.exists():
            with events_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        await websocket.send_text(line)
                last_pos = events_file.stat().st_size

        # Tail for new events
        heartbeat_counter = 0
        max_iterations = 3600  # ~30 minutes at 0.5s intervals

        for _ in range(max_iterations):
            await asyncio.sleep(0.5)  # Check every 500ms

            # Check for new events
            if events_file.exists():
                current_size = events_file.stat().st_size
                if current_size > last_pos:
                    with events_file.open("r", encoding="utf-8") as f:
                        f.seek(last_pos)
                        new_content = f.read()
                        for line in new_content.strip().split("\n"):
                            if line:
                                await websocket.send_text(line)
                                # Check if run completed
                                try:
                                    event = json.loads(line)
                                    if event.get("type") in (
                                        "run_completed",
                                        "run_failed",
                                        "run_cancelled",
                                    ):
                                        logger.info(f"Run {run_id} completed, closing WebSocket")
                                        await websocket.close(code=1000)
                                        return
                                except json.JSONDecodeError:
                                    pass
                        last_pos = current_size

            # Send heartbeat every 10 iterations (5 seconds)
            heartbeat_counter += 1
            if heartbeat_counter >= 10:
                heartbeat_counter = 0
                try:
                    await websocket.send_json({"type": "heartbeat", "run_id": run_id})
                except Exception:
                    break  # Connection closed

            # Check if run is complete via manifest (fallback)
            if manifest_file.exists():
                try:
                    manifest = json.loads(manifest_file.read_text())
                    status = manifest.get("status", "")
                    if status in ("COMPLETED", "FAILED", "CANCELLED", "STALE"):
                        # Send final status event if not already sent via events file
                        await websocket.send_json(
                            {
                                "type": "run_completed",
                                "run_id": run_id,
                                "status": status,
                                "source": "manifest_fallback",
                            }
                        )
                        logger.info(f"Run {run_id} status is {status}, closing WebSocket")
                        await websocket.close(code=1000)
                        return
                except (json.JSONDecodeError, FileNotFoundError):
                    pass

        # Timeout reached
        logger.warning(f"WebSocket timeout for run {run_id}")
        await websocket.close(code=4008)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for run {run_id}")
    except Exception as e:
        logger.error(f"WebSocket error for run {run_id}: {e}")
        try:
            await websocket.close(code=4500)
        except Exception:
            pass


@router.get("/api/runs/{run_id}/events")
async def get_run_events(run_id: str):
    """
    Get all events for a run (REST fallback for non-WebSocket clients).

    Returns a list of all events that have been emitted for the run.
    """
    run_dir = _get_run_dir(run_id)
    if not run_dir:
        return {"error": "Run not found", "events": []}

    events_file = _get_events_file(run_dir)
    if not events_file.exists():
        return {"events": []}

    events = []
    with events_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    return {"events": events}
