from pathlib import Path
import json
import logging
from typing import List, Dict, Any, Optional
import time
import os

logger = logging.getLogger("server.index")

def _runs_root() -> Path:
    return Path(os.environ.get("MODEL_LAB_RUNS_ROOT", "runs")).resolve()

class RunsIndex:
    _instance: Optional['RunsIndex'] = None
    _transcript_cache: Dict[str, Dict[str, Any]] = {} # run_id -> {mtime: int, dto: Dict}

    def __init__(self):
        self._cache: List[Dict[str, Any]] = []
        self._last_updated: float = 0
        self._ttl: int = 5  # seconds

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def list_runs(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        # Auto-refresh if TTL expired
        if force_refresh or (time.time() - self._last_updated > self._ttl):
            self._refresh_index()
        return self._cache

    def refresh(self) -> List[Dict[str, Any]]:
        """Force a refresh of the index immediately."""
        self._refresh_index()
        return self._cache

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        # Refresh if not found or stale
        if not any(r["run_id"] == run_id for r in self._cache):
            self._refresh_index()
        
        for r in self._cache:
            if r["run_id"] == run_id:
                return r
        return None

    def get_transcript(self, run_id: str) -> Optional[Dict[str, Any]]:
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

    def _reload_transcript(self, run_id: str, run: Dict, manifest_path: Path, mtime: int) -> Optional[Dict[str, Any]]:
        logger.debug(f"Reloading transcript for {run_id}")
        try:
            # Re-read manifest to find artifacts
            manifest = json.loads(manifest_path.read_text())
            run_dir = manifest_path.parent
            
            segments = []
            chapters = [] # TODO: Load chapters if available
            
            # Locate ASR artifact
            asr_step = manifest.get("steps", {}).get("asr", {})
            asr_artifacts = asr_step.get("artifacts", [])
            
            asr_path = None
            for art in asr_artifacts:
                if isinstance(art, dict) and art.get("path", "").endswith(".json"):
                    asr_path = art.get("path")
                    break
            
            # Fallback path logic
            target_file = None
            if asr_path:
                target_file = run_dir / asr_path
            else:
                fallback = run_dir / "artifacts/asr.json"
                if fallback.exists():
                    target_file = fallback
            
            if target_file and target_file.exists():
                raw_asr = json.loads(target_file.read_text())
                if "segments" in raw_asr:
                    # Normalize
                    for i, seg in enumerate(raw_asr["segments"]):
                        start = seg.get("start")
                        end = seg.get("end")
                        text = seg.get("text", "").strip()
                        speaker = seg.get("speaker")
                        
                        # Assign stable ID if not present (using index fallback)
                        # Ideally should use start time or hash, but index is okay for V2 immutable-ish transcripts
                        seg_id = f"seg_{i}" 
                        
                        segments.append({
                            "id": seg_id,
                            "start_s": start,
                            "end_s": end,
                            "text": text,
                            "speaker": speaker
                        })

            # Load Chapters
            chapters_step = manifest.get("steps", {}).get("chapters", {})
            chapters_artifacts = chapters_step.get("artifacts", [])
            
            chapters_path = None
            for art in chapters_artifacts:
                if isinstance(art, dict) and art.get("path", "").endswith(".json"):
                    chapters_path = art.get("path")
                    break
            
            # Fallback
            chapters_file = None
            if chapters_path:
                chapters_file = run_dir / chapters_path
            else:
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
                            
                            chapters.append({
                                "start_s": start,
                                "end_s": end,
                                "title": title
                            })
                        
                        # Sort by start_s
                        chapters.sort(key=lambda x: x["start_s"])
                except Exception as e:
                    logger.warning(f"Failed to load chapters for {run_id}: {e}")

            dto = {
                "run_id": run_id,
                "segments": segments,
                "chapters": chapters
            }
            
            # Update cache
            self._transcript_cache[run_id] = {
                "mtime": mtime,
                "dto": dto
            }
            return dto

        except Exception as e:
            logger.error(f"Failed to load transcript for {run_id}: {e}")
            return None

    def search_run(self, run_id: str, query: str, limit: int = 50) -> Dict[str, Any]:
        """
        Search within a run's transcript.
        """
        # Constraints
        query = query.strip()
        if len(query) < 2:
            return {"query": query, "results": []}
        
        limit = min(limit, 200) # Hard cap
        
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
                results.append({
                    "segment_id": seg["id"],
                    "start_s": seg["start_s"],
                    "end_s": seg["end_s"],
                    "text": text,
                    "match_start": idx,
                    "match_end": idx + len(query)
                })
                
                if len(results) >= limit:
                    break
        
        return {
            "query": query,
            "results": results
        }

    def _refresh_index(self):
        """Atomic refresh of the index."""
        logger.debug("Refreshing runs index...")
        runs = self._scan_runs()
        
        # Atomic swap (assignment is atomic in Python)
        self._cache = runs
        self._last_updated = time.time()
        logger.debug(f"Indexed {len(runs)} runs.")

    def _scan_runs(self) -> List[Dict[str, Any]]:
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
                
                # Extract summary fields
                run_summary = {
                    "run_id": data.get("run_id", p.parent.name),
                    "status": data.get("status", "UNKNOWN"),
                    "started_at": data.get("started_at"),
                    "input_filename": Path(data.get("input_path", "")).name, # Extract filename only
                    "duration": data.get("duration_s"),
                    "steps_completed": list(data.get("steps", {}).keys()),
                    "manifest_path": str(p),
                    # We might store absolute path to root for safe_files optimization
                    "root_path": str(p.parent)
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
