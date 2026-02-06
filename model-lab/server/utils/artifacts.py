"""Artifact resolution utilities for server API layer.

This module provides server-side artifact path resolution with fallback logic
for different bundle formats (Meeting Pack V1, legacy paths).
"""

from pathlib import Path
from typing import Any


def resolve_artifact_relpaths(artifact: str) -> list[str]:
    """Get ordered list of relative paths to try for an artifact.

    Args:
        artifact: Artifact name ('transcript', 'summary', 'action_items', 'decisions')

    Returns:
        List of relative paths in priority order (bundle first, then legacy)
    """
    if artifact == "transcript":
        return [
            "bundle/transcript.json",  # Meeting Pack canonical
            "bundle/transcript.txt",  # Legacy bundle
            "artifacts/transcript.txt",  # Simple run output
            "asr/transcript.txt",  # Task-specific output
        ]
    elif artifact == "summary":
        return [
            "bundle/summary.md",
            "session/summary.md",  # Legacy
        ]
    elif artifact == "action_items":
        return [
            "bundle/action_items.csv",
            "session/action_items.csv",  # Legacy
        ]
    elif artifact == "decisions":
        return [
            "bundle/decisions.md",
        ]
    else:
        return []


def read_artifact_text(run_dir: Path, artifact: str, max_bytes: int = 200_000) -> dict[str, Any]:
    """Read artifact text with size checks and fallback resolution.

    Args:
        run_dir: Run directory root
        artifact: Artifact name
        max_bytes: Maximum file size to read

    Returns:
        Dict with keys:
            - available: bool
            - text: str | None
            - size: int (bytes if available)
            - truncated: bool (if file too large)
            - error: str | None ('PREVIEW_TOO_LARGE' | 'READ_ERROR')
            - rel_path: str | None (which path resolved)
    """
    candidates = resolve_artifact_relpaths(artifact)

    # Find first existing path
    target_path = None
    rel_path = None
    for rel in candidates:
        p = run_dir / rel
        if p.exists():
            target_path = p
            rel_path = rel
            break

    if not target_path:
        return {"available": False, "text": None}

    # Check size before reading
    file_size = target_path.stat().st_size
    if file_size > max_bytes:
        return {
            "available": True,
            "text": None,
            "truncated": True,
            "size": file_size,
            "error": "PREVIEW_TOO_LARGE",
            "rel_path": rel_path,
        }

    # Read text
    try:
        text = target_path.read_text(encoding="utf-8")
        return {
            "available": True,
            "text": text,
            "size": len(text),
            "rel_path": rel_path,
        }
    except Exception:
        return {
            "available": False,
            "text": None,
            "error": "READ_ERROR",
            "rel_path": rel_path,
        }
