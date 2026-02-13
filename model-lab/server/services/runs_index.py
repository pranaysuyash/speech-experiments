import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("server.index")


def _compute_config_hash(config: dict[str, Any]) -> str:
    """Compute a stable hash of pipeline config for grouping similar runs."""
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def _runs_root() -> Path:
    return Path(os.environ.get("MODEL_LAB_RUNS_ROOT", "runs")).resolve()


class RunsIndex:
    _instance: Optional["RunsIndex"] = None
    _transcript_cache: dict[str, dict[str, Any]] = {}  # run_id -> {mtime: int, dto: Dict}

    def __init__(self):
        self._cache: list[dict[str, Any]] = []
        self._last_updated: float = 0
        self._ttl: int = 5  # seconds

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def list_runs(self, force_refresh: bool = False) -> list[dict[str, Any]]:
        # Auto-refresh if TTL expired
        if force_refresh or (time.time() - self._last_updated > self._ttl):
            self._refresh_index()
        return self._cache

    def refresh(self) -> list[dict[str, Any]]:
        """Force a refresh of the index immediately."""
        self._refresh_index()
        return self._cache

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        # Refresh if not found or stale
        if not any(r["run_id"] == run_id for r in self._cache):
            self._refresh_index()

        for r in self._cache:
            if r["run_id"] == run_id:
                return r
        return None

    def get_transcript(self, run_id: str) -> dict[str, Any] | None:
        """
        Returns cached normalized transcript DTO.
        Reloads if manifest mtime has changed.
        """
        run = self.get_run(run_id)
        if not run:
            return None

        # Check raw manifest mtime for cache invalidation
        # We use the manifest path from the index
        manifest_path = Path(run["manifest_path"])
        if not manifest_path.exists():
            return None

        current_mtime = manifest_path.stat().st_mtime_ns

        # Check cache
        cached = self._transcript_cache.get(run_id)
        if cached and cached["mtime"] == current_mtime:
            return cached["dto"]

        # Cache miss or stale - Reload
        return self._reload_transcript(run_id, run, manifest_path, current_mtime)

    def _reload_transcript(
        self, run_id: str, run: dict, manifest_path: Path, mtime: int
    ) -> dict[str, Any] | None:
        logger.debug(f"Reloading transcript for {run_id}")
        try:
            # Re-read manifest to find artifacts
            manifest = json.loads(manifest_path.read_text())
            try:
                schema_version = int(manifest.get("manifest_schema_version", 1))
            except (ValueError, TypeError):
                schema_version = 1

            registry = manifest.get("artifacts_by_type", {})
            run_dir = manifest_path.parent

            segments = []
            chapters = []

            # --- Artifact Resolution (Schema Gating) ---
            asr_path = None

            if schema_version >= 2:
                # v2: MUST be in registry
                candidates = registry.get("asr", []) + registry.get("transcript", [])
                # Prefer transcript (alignment) over asr?
                # Actually we want 'alignment' result usually for speakers.
                # Let's check 'alignment' first, then 'asr'.
                # But logic below parses 'asr.json' format (segments).
                # 'alignment' step produces 'alignment.json' which has segments.

                # Let's look for 'alignment' type first (better quality), then 'asr'.
                target_candidates = registry.get("alignment", []) or registry.get("asr", [])

                if not target_candidates:
                    raise RuntimeError(
                        f"E_ARTIFACT_REGISTRY_MISSING: No 'alignment' or 'asr' artifacts in registry (Schema v{schema_version})"
                    )

                # Sort by mtime descending (latest)
                target_candidates.sort(key=lambda x: x.get("mtime", 0), reverse=True)
                asr_path = target_candidates[0].get("path")
            else:
                # v1: Step traversal fallback (Legacy)
                asr_step = manifest.get("steps", {}).get("asr", {})
                asr_artifacts = asr_step.get("artifacts", [])
                for art in asr_artifacts:
                    if isinstance(art, dict) and art.get("path", "").endswith(".json"):
                        asr_path = art.get("path")
                        break

            # Fallback path logic (Legacy filesystem probe - allow for v1 only? or keep for robustness?)
            # Validating "E_ARTIFACT_REGISTRY_MISSING" for v2 implies we should FAIL if v2 and not in registry.
            # But the 'target_candidates' check above already ensures that.

            target_file = None
            if asr_path:
                candidate_path = Path(asr_path)
                target_file = candidate_path if candidate_path.is_absolute() else run_dir / asr_path
            elif schema_version < 2:
                # Only probe filesystem in v1
                fallback = run_dir / "artifacts/asr.json"
                if fallback.exists():
                    target_file = fallback

            bundle_transcript = run_dir / "bundle" / "transcript.json"
            if bundle_transcript.exists():
                raw_bundle = json.loads(bundle_transcript.read_text())
                bundle_segments = raw_bundle.get("segments", [])
                if isinstance(bundle_segments, list) and bundle_segments:
                    for i, seg in enumerate(bundle_segments):
                        start = (
                            seg.get("start_s")
                            if seg.get("start_s") is not None
                            else seg.get("start")
                        )
                        end = seg.get("end_s") if seg.get("end_s") is not None else seg.get("end")
                        text = seg.get("text", "").strip()
                        speaker = seg.get("speaker")
                        segments.append(
                            {
                                "id": f"bundle_seg_{i}",
                                "start_s": start,
                                "end_s": end,
                                "text": text,
                                "speaker": speaker,
                            }
                        )
            elif target_file and target_file.exists():
                raw_asr = json.loads(target_file.read_text())
                segment_list = raw_asr.get("segments")
                if not segment_list:
                    segment_list = (raw_asr.get("output") or {}).get("segments")
                if not segment_list:
                    segment_list = (raw_asr.get("result") or {}).get("segments")

                if isinstance(segment_list, list):
                    for i, seg in enumerate(segment_list):
                        start = seg.get("start")
                        end = seg.get("end")
                        text = seg.get("text", "").strip()
                        speaker = seg.get("speaker")
                        segments.append(
                            {
                                "id": f"seg_{i}",
                                "start_s": start,
                                "end_s": end,
                                "text": text,
                                "speaker": speaker,
                            }
                        )

            # Load Chapters (Schema Gated)
            chapters_path = None
            if schema_version >= 2:
                candidates = registry.get("chapters", [])
                if candidates:
                    candidates.sort(key=lambda x: x.get("mtime", 0), reverse=True)
                    chapters_path = candidates[0].get("path")
            else:
                chapters_step = manifest.get("steps", {}).get("chapters", {})
                for art in chapters_step.get("artifacts", []):
                    if isinstance(art, dict) and art.get("path", "").endswith(".json"):
                        chapters_path = art.get("path")
                        break

            chapters_file = None
            if chapters_path:
                chapters_file = run_dir / chapters_path
            elif schema_version < 2:
                fallback = run_dir / "artifacts/chapters.json"
                if fallback.exists():
                    chapters_file = fallback

            if chapters_file and chapters_file.exists():
                try:
                    raw_chapters = json.loads(chapters_file.read_text())
                    # Chapters schema: {"chapters": [{start_s, end_s, title}]}
                    if "chapters" in raw_chapters:
                        # Normalize: sort, clamp, validate
                        duration = segments[-1]["end_s"] if segments else 0
                        for ch in raw_chapters["chapters"]:
                            start = ch.get("start_s", 0)
                            end = ch.get("end_s", 0)
                            title = ch.get("title", "").strip()

                            # Validate and clamp
                            if start < 0:
                                start = 0
                            if end > duration:
                                end = duration
                            if start >= end or not title:
                                continue  # Skip invalid

                            chapters.append({"start_s": start, "end_s": end, "title": title})

                        # Sort by start_s
                        chapters.sort(key=lambda x: x["start_s"])
                except Exception as e:
                    logger.warning(f"Failed to load chapters for {run_id}: {e}")

            dto = {"run_id": run_id, "segments": segments, "chapters": chapters}

            # Update cache
            self._transcript_cache[run_id] = {"mtime": mtime, "dto": dto}
            return dto

        except Exception as e:
            if "E_ARTIFACT_REGISTRY_MISSING" in str(e):
                raise e
            logger.error(f"Failed to load transcript for {run_id}: {e}")
            return None

    def search_run(self, run_id: str, query: str, limit: int = 50) -> dict[str, Any]:
        """
        Search within a run's transcript.
        """
        # Constraints
        query = query.strip()
        if len(query) < 2:
            return {"query": query, "results": []}

        limit = min(limit, 200)  # Hard cap

        dto = self.get_transcript(run_id)
        if not dto:
            return {"query": query, "results": []}

        query_lower = query.lower()
        results = []

        for seg in dto["segments"]:
            text = seg["text"]
            text_lower = text.lower()
            idx = text_lower.find(query_lower)

            if idx != -1:
                results.append(
                    {
                        "segment_id": seg["id"],
                        "start_s": seg["start_s"],
                        "end_s": seg["end_s"],
                        "text": text,
                        "match_start": idx,
                        "match_end": idx + len(query),
                    }
                )

                if len(results) >= limit:
                    break

        return {"query": query, "results": results}

    def _refresh_index(self):
        """Atomic refresh of the index."""
        logger.debug("Refreshing runs index...")
        runs = self._scan_runs()

        # Atomic swap (assignment is atomic in Python)
        self._cache = runs
        self._last_updated = time.time()
        logger.debug(f"Indexed {len(runs)} runs.")

    def _scan_runs(self) -> list[dict[str, Any]]:
        """Scans the filesystem for runs. Returns list."""
        runs = []
        # Crawl manifest.json files
        # Expecting: runs/sessions/<input_hash>/<run_id>/manifest.json
        manifests = _runs_root().glob("sessions/*/*/manifest.json")

        for p in manifests:
            try:
                # Read minimal info
                # To be fast, maybe we don't parse full JSON if it's huge?
                # But typically manifests are small enough.
                data = json.loads(p.read_text())

                # Extract input hash from run_request.json if available
                input_hash = None
                run_request_path = p.parent / "run_request.json"
                preprocessing_ops: list[str] = []
                custom_steps: list[str] | None = None
                template_used: str | None = None

                if run_request_path.exists():
                    try:
                        run_request = json.loads(run_request_path.read_text())
                        input_hash = run_request.get("sha256")
                        run_request.get("pipeline_config", {})
                        preprocessing_ops = run_request.get("preprocessing", []) or []
                        custom_steps = run_request.get("steps_custom")
                        template_used = run_request.get("pipeline_template")
                    except Exception:
                        pass

                # Compute pipeline config hash for grouping
                config_for_hash = {
                    "steps": data.get("steps_requested") or list(data.get("steps", {}).keys()),
                    "preprocessing": preprocessing_ops,
                    "config": data.get("config", {}),
                }
                pipeline_config_hash = _compute_config_hash(config_for_hash)

                # Extract summary fields
                run_summary = {
                    "run_id": data.get("run_id", p.parent.name),
                    "status": data.get("status", "UNKNOWN"),
                    "started_at": data.get("started_at"),
                    "created_at": data.get("started_at"),  # Alias for sorting
                    "input_filename": Path(
                        data.get("input_path", "")
                    ).name,  # Extract filename only
                    "duration": data.get("duration_s"),
                    "steps_completed": list(data.get("steps", {}).keys()),
                    "manifest_path": str(p),
                    # We might store absolute path to root for safe_files optimization
                    "root_path": str(p.parent),
                    # New fields for run history & comparison
                    "input_hash": input_hash,
                    "preprocessing_ops": preprocessing_ops,
                    "custom_steps": custom_steps,
                    "template_used": template_used,
                    "pipeline_config_hash": pipeline_config_hash,
                }
                runs.append(run_summary)
            except Exception as e:
                logger.warning(f"Failed to index manifest {p}: {e}")

        # Sort by started_at desc
        runs.sort(key=lambda x: x.get("started_at") or "", reverse=True)
        return runs


# Singleton accessor
def get_index() -> RunsIndex:
    return RunsIndex.get_instance()


def cleanup_old_runs(retention_days: int = 30, dry_run: bool = False) -> dict[str, Any]:
    """
    Clean up runs older than retention_days.

    Args:
        retention_days: Delete runs not modified in this many days
        dry_run: If True, only report what would be deleted

    Returns:
        {"deleted": [run_ids], "freed_bytes": int, "errors": [messages]}
    """
    import shutil
    from datetime import datetime, timedelta

    runs_root = _runs_root()
    sessions_dir = runs_root / "sessions"

    if not sessions_dir.exists():
        return {"deleted": [], "freed_bytes": 0, "errors": ["No sessions directory found"]}

    cutoff_time = datetime.now() - timedelta(days=retention_days)
    deleted = []
    freed_bytes = 0
    errors = []

    # Find all run directories
    run_dirs = list(sessions_dir.glob("*/**/"))

    for run_dir in run_dirs:
        # Check if it's a run directory (has manifest.json)
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        try:
            # Check modification time
            mtime = datetime.fromtimestamp(manifest_path.stat().st_mtime)
            if mtime > cutoff_time:
                continue  # Skip runs newer than cutoff

            # Calculate size
            total_size = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file())

            if dry_run:
                deleted.append(run_dir.name)
            else:
                # Delete the run directory
                shutil.rmtree(run_dir)
                deleted.append(run_dir.name)
                freed_bytes += total_size
                logger.info(f"Deleted run {run_dir.name}, freed {total_size} bytes")

        except Exception as e:
            errors.append(f"Failed to delete {run_dir.name}: {e}")

    return {
        "deleted": deleted,
        "freed_bytes": freed_bytes,
        "errors": errors,
        "retention_days": retention_days,
        "dry_run": dry_run,
    }


def get_disk_usage() -> dict[str, Any]:
    """
    Get disk usage for runs directory.

    Returns:
        {"total_bytes": int, "used_bytes": int, "free_bytes": int, "run_count": int}
    """
    import shutil

    runs_root = _runs_root()

    if not runs_root.exists():
        return {"total_bytes": 0, "used_bytes": 0, "free_bytes": 0, "run_count": 0}

    # Get disk usage
    usage = shutil.disk_usage(runs_root)

    # Count runs
    sessions_dir = runs_root / "sessions"
    run_count = 0
    if sessions_dir.exists():
        run_count = sum(1 for _ in sessions_dir.glob("*/**/manifest.json"))

    return {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "run_count": run_count,
        "runs_root": str(runs_root),
    }
