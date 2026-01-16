from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

from harness.media_ingest import IngestConfig, ingest_media, sha256_file

logger = logging.getLogger("session")

Status = str  # "PENDING" | "RUNNING" | "COMPLETED" | "FAILED"

@dataclass
class SessionContext:
    input_path: Path
    output_dir: Path
    artifacts_dir: Path
    logs_dir: Path
    bundle_dir: Path
    
    # Store ingest result here
    ingest: Optional[Dict[str, Any]] = None

    @property
    def audio_path(self) -> Optional[Path]:
        if not self.ingest:
            return None
        return Path(self.ingest["processed_audio_path"])

    @property
    def audio_fingerprint(self) -> Optional[str]:
        if not self.ingest:
            return None
        return self.ingest["audio_fingerprint"]


@dataclass
class StepDef:
    name: str
    deps: List[str]
    func: Callable[[SessionContext], Dict[str, Any]]
    # Used for resume checks
    artifact_paths: Callable[[Dict[str, Any]], List[Path]]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(tmp), str(path))


def compute_run_id(input_hash: str) -> str:
    # Deterministic enough: timestamp + short hash
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    short = hashlib.sha256((ts + input_hash).encode("utf-8")).hexdigest()[:10]
    return f"{ts}_{short}"

# Schema version for Manifest
MANIFEST_SCHEMA_VERSION = "1.2.0"

class SessionRunner:
    def __init__(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        *,
        force: bool = False,
        resume: bool = True,
        preprocessing: Optional[IngestConfig] = None,
        embedding_cache_dir: Optional[Path] = None, # Kept for existing logic support (though moving towards context)
        steps: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None # Generic config override
    ) -> None:
        self.input_path = Path(input_path).resolve()
        self.base_output_dir = Path(output_dir).resolve()
        self.force = force
        self.resume = resume
        self.preprocessing = preprocessing or IngestConfig()
        self.requested_steps = steps
        self.embedding_cache_dir = embedding_cache_dir
        self.extra_config = config or {}

        if not self.input_path.exists():
             raise FileNotFoundError(f"Input file not found: {self.input_path}")

        self.input_hash = sha256_file(self.input_path)
        
        # If resuming, we might need to find the latest run_id (or provided one)
        # But per new design, we compute deterministic run_id or use existing logic?
        # User skeleton uses timestamped run_id. Let's stick to that.
        # Ideally, passed `resume_from` would override this.
        # For now, let's implement basic directory structure.
        
        # NOTE: logic to find latest run or specific run needs to be here if we want strict resume.
        # But compute_run_id uses timestamp. So essentially every run is new?
        # User says: "Resume: skip only when hashes match and artifacts exist."
        # This implies we are resuming *within* a run directory that is passed?
        # OR we scan for a matching run?
        # CLI usually takes --resume-from PATH.
        # If we just run, we likely create a new run_id.
        # Wait, if run_id is timestamped, resume is impossible unless we pass the ID.
        # Let's support `resume_from` via `extra_config` or separate arg if needed later.
        # For now, adopting user's skeleton which just computes a new ID.
        # BUT if we want to resume, we probably want to point to an existing dir.
        
        # Let's adapt: If the CLI passes a resume path, we use it.
        # The CLI currently uses --resume-from. We should respect that if passed.
        # Let's handle it properly.
        
        resume_from = self.extra_config.get("resume_from")
        if resume_from:
             self.session_dir = Path(resume_from).resolve()
             self.run_id = self.session_dir.name
             # We assume input hash matches path structure? Not strictly required but good hygiene.
        else:
             self.run_id = compute_run_id(self.input_hash)
             self.session_dir = self.base_output_dir / "sessions" / self.input_hash / self.run_id

        self.manifest_path = self.session_dir / "manifest.json"

        self.ctx = SessionContext(
            input_path=self.input_path,
            output_dir=self.session_dir,
            artifacts_dir=self.session_dir / "artifacts",
            logs_dir=self.session_dir / "logs",
            bundle_dir=self.session_dir / "bundle",
        )
        
        # Inject embedding cache into context if needed by legacy steps
        # Ideally steps should take config from context.
        
        self.steps: Dict[str, StepDef] = {}

    def _init_dirs(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.ctx.logs_dir.mkdir(parents=True, exist_ok=True)
        self.ctx.bundle_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure File Logging
        log_file = self.ctx.logs_dir / "run.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        # Attach to root logger? Or specific loggers?
        # Root logger captures everything.
        root = logging.getLogger()
        # Avoid duplicate handlers
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file) for h in root.handlers):
            root.addHandler(file_handler)

    def _default_manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": self.run_id,
            "input": {
                "original_path": str(self.input_path),
                "input_hash": self.input_hash,
            },
            "config": {
                "steps": self.requested_steps or [],
                "preprocessing": dataclasses.asdict(self.preprocessing),
                "embedding_cache_dir": str(self.embedding_cache_dir) if self.embedding_cache_dir else None
            },
            "status": "PENDING",
            "started_at": None,
            "ended_at": None,
            "duration_ms": None,
            "steps": {},
            "final": {},
            "warnings": [],
        }

    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return self._default_manifest()

    def _save_manifest(self, m: Dict[str, Any]) -> None:
        atomic_write_json(self.manifest_path, m)

    def _write_eval_json(self, manifest: Dict[str, Any]) -> None:
        """Write eval.json contract for results/findings UI (V1: identity only)."""
        bundle_dir = self.session_dir / "bundle"
        if not bundle_dir.exists():
            return  # No bundle, skip eval
        
        eval_path = bundle_dir / "eval.json"
        
        # Build minimal eval with identity and availability checks
        from datetime import datetime
        
        eval_data = {
            "schema_version": "1",
            "run_id": manifest["run_id"],
            "use_case_id": None,  # V1: not tracked yet
            "model_id": None,     # V1: not tracked yet
            "params": manifest.get("config", {}),
            "metrics": {},  # V1: no metrics computed yet
            "checks": [
                {
                    "name": "eval_availability",
                    "passed": True,
                    "severity": "info",
                    "message": "Evaluation contract available (V1: identity only)",
                    "evidence_paths": []
                }
            ],
            "findings": [],  # V1: no findings yet
            "generated_at": datetime.now().isoformat()
        }
        
        # Add check for missing metrics
        if manifest["status"] == "COMPLETED":
            eval_data["checks"].append({
                "name": "metrics_availability",
                "passed": False,
                "severity": "warn",
                "message": "Metrics computation not yet implemented",
                "evidence_paths": []
            })
        
        atomic_write_json(eval_path, eval_data)

    def _register_steps(self) -> None:
        # 1. Ingest
        def ingest_func(ctx: SessionContext) -> Dict[str, Any]:
            return ingest_media(ctx.input_path, ctx.artifacts_dir, self.preprocessing)

        def ingest_artifacts(step_result: Dict[str, Any]) -> List[Path]:
            return [Path(step_result["processed_audio_path"])]

        self.steps["ingest"] = StepDef(
            name="ingest",
            deps=[],
            func=ingest_func,
            artifact_paths=ingest_artifacts,
        )
        
        # 2. ASR (Wire up existing harness)
        def asr_func(ctx: SessionContext) -> Dict[str, Any]:
             if not ctx.audio_path or not ctx.audio_path.exists():
                 raise RuntimeError("Missing audio path for ASR")
             # Call harness function. Assuming refactoring to take (input, output_dir, config)
             # Current: run_asr(input_path, output_dir, model_name, device, logic_config)
             # We should probably pass the audio_path as input.
             # And we need to construct a robust output.
             # For now, let's assume we wrap it to behave.
             # TODO: Refactor harness/asr.py run_asr to be friendlier or wrap loosely.
             # This assumes ASR runner writes to `artifacts/asr.json`
             # We'll pass ctx.artifacts_dir and let it write there.
             from harness.asr import run_asr
             return run_asr(ctx.audio_path, ctx.artifacts_dir, config=self.extra_config.get("asr", {}))

        def asr_artifacts(step_result: Dict[str, Any]) -> List[Path]:
             # Assuming step_result contains path or we know it
             return [Path(p["path"]) for p in step_result.get("artifacts", [])]

        self.steps["asr"] = StepDef(
            name="asr",
            deps=["ingest"],
            func=asr_func,
            artifact_paths=asr_artifacts
        )
        
        # 3. Diarization
        def diarization_func(ctx: SessionContext) -> Dict[str, Any]:
            if not ctx.audio_path: raise RuntimeError("No audio")
            from harness.diarization import run_diarization
            return run_diarization(ctx.audio_path, ctx.artifacts_dir, config=self.extra_config.get("diarization", {}))
            
        def diarization_artifacts(res: Dict) -> List[Path]:
            return [Path(p["path"]) for p in res.get("artifacts", [])]
            
        self.steps["diarization"] = StepDef(
            name="diarization",
            deps=["ingest"],
            func=diarization_func,
            artifact_paths=diarization_artifacts
        )
        
        # 4. Alignment
        def alignment_func(ctx: SessionContext) -> Dict[str, Any]:
            # Needs ASR and Diarization artifacts.
            # We assume they are in ctx.artifacts_dir with standard names?
            # Or we fetch them from manifest result?
            # Ideally step_result should be accessible.
            # For simplicity, relying on standard file paths in artifacts_dir for now,
            # but stricter way is to lookup from dependency result (Context enrichment).
            # But run_alignment currently looks for files.
            asr_path = ctx.artifacts_dir / "asr.json"
            diar_path = ctx.artifacts_dir / "diarization.json"
            from harness.alignment import run_alignment
            return run_alignment(asr_path, diar_path, ctx.artifacts_dir)

        def alignment_artifacts(res: Dict) -> List[Path]:
             return [Path(p["path"]) for p in res.get("artifacts", [])]
             
        self.steps["alignment"] = StepDef(
            name="alignment",
            deps=["asr", "diarization"],
            func=alignment_func,
            artifact_paths=alignment_artifacts
        )
        
        # 5. Chapters
        def chapters_func(ctx: SessionContext) -> Dict[str, Any]:
            align_path = ctx.artifacts_dir / "alignment.json"
            from harness.chapters import run_chapters
            return run_chapters(align_path, ctx.artifacts_dir, embedding_cache_dir=self.embedding_cache_dir)
            
        def chapters_artifacts(res: Dict) -> List[Path]:
            # Returns list of artifacts (chapters.json)
            return [Path(p["path"]) for p in res.get("artifacts", [])]
            
        self.steps["chapters"] = StepDef(
            name="chapters",
            deps=["alignment"],
            func=chapters_func,
            artifact_paths=chapters_artifacts
        )
        
        # 6. NLP (Summarize)
        def summarize_func(ctx: SessionContext) -> Dict[str, Any]:
             align_path = ctx.artifacts_dir / "alignment.json"
             from harness.nlp import run_summarize_by_speaker
             return run_summarize_by_speaker(align_path, ctx.artifacts_dir)
        
        def summarize_artifacts(res: Dict) -> List[Path]:
            return [Path(p["path"]) for p in res.get("artifacts", [])]
            
        self.steps["summarize_by_speaker"] = StepDef(
            name="summarize_by_speaker",
            deps=["alignment"],
            func=summarize_func,
            artifact_paths=summarize_artifacts
        )
        
        # 7. NLP (Action Items)
        def action_items_func(ctx: SessionContext) -> Dict[str, Any]:
             align_path = ctx.artifacts_dir / "alignment.json"
             from harness.nlp import run_action_items
             return run_action_items(align_path, ctx.artifacts_dir)

        def action_items_artifacts(res: Dict) -> List[Path]:
             return [Path(p["path"]) for p in res.get("artifacts", [])]

        self.steps["action_items_assignee"] = StepDef(
            name="action_items_assignee",
            deps=["alignment"],
            func=action_items_func,
            artifact_paths=action_items_artifacts
        )

        # 8. Bundle (Meeting Pack exports)
        def bundle_func(ctx: SessionContext) -> Dict[str, Any]:
            from harness.meeting_pack import build_meeting_pack
            return build_meeting_pack(ctx.output_dir)

        def bundle_artifacts(res: Dict[str, Any]) -> List[Path]:
            paths = []
            for p in res.get("written_paths", []):
                if isinstance(p, str) and p:
                    paths.append(Path(p))
            return paths

        self.steps["bundle"] = StepDef(
            name="bundle",
            deps=sorted([k for k in self.steps.keys() if k != "bundle"]),
            func=bundle_func,
            artifact_paths=bundle_artifacts,
        )

    def _step_entry(self, m: Dict[str, Any], step: str) -> Dict[str, Any]:
        return m["steps"].setdefault(step, {"status": "PENDING", "artifacts": [], "warnings": []})

    def _artifact_exists_and_hash_matches(self, artifact_path: Path, expected_hash: Optional[str]) -> bool:
        if not artifact_path.exists():
            return False
        if not expected_hash:
            return False
        actual = sha256_file(artifact_path)
        return actual == expected_hash

    def _is_step_valid_for_resume(self, m: Dict[str, Any], step_name: str, step_def: StepDef) -> bool:
        entry = m["steps"].get(step_name)
        if not entry or entry.get("status") != "COMPLETED":
            return False

        # For ingest specifically, enforce strict config + source hash match.
        if step_name == "ingest":
            stored = entry.get("result", {})
            if stored.get("source_media_hash") != sha256_file(self.input_path):
                return False

            # preprocess_hash depends on config+ffmpeg version, so comparing preprocess_hash is enough.
            # But we also ensure the stored config matches for transparency.
            stored_cfg = stored.get("preprocessing_config")
            if stored_cfg != dataclasses.asdict(self.preprocessing):
                return False

            processed_path = Path(stored.get("processed_audio_path", ""))
            if not processed_path.exists():
                return False

            # Validate stored audio_content_hash matches the file
            expected_hash = stored.get("audio_content_hash")
            if not self._artifact_exists_and_hash_matches(processed_path, expected_hash):
                return False

            return True

        # Generic step validation: check all recorded artifacts exist and hashes match
        # Artifacts here are {type, path, hash}
        artifacts = entry.get("artifacts", [])
        if not artifacts:
            return False

        for a in artifacts:
            # path is likely relative or absolute.
            # session runner stored paths in manifest.
            p = Path(a["path"])
            if not p.is_absolute():
                 p = self.session_dir / p
            
            h = a.get("hash")
            if not self._artifact_exists_and_hash_matches(p, h):
                return False

        return True

    def _invalidate_downstream(self, m: Dict[str, Any], step_name: str) -> None:
        # Mark dependents PENDING and drop artifact hashes
        dependents = self._collect_dependents(step_name)
        for dep in dependents:
            if dep == step_name:
                continue
            entry = m["steps"].get(dep)
            if not entry:
                continue
            entry["status"] = "PENDING"
            entry["artifacts"] = []
            entry.pop("result", None)
            entry.pop("metrics", None)
            entry["warnings"] = []
            # Also clear ended_at etc?
            entry["ended_at"] = None

    def _collect_dependents(self, step_name: str) -> List[str]:
        # Simple graph walk
        out: List[str] = []
        visited = set()

        def dfs(s: str) -> None:
            for name, sd in self.steps.items():
                if s in sd.deps and name not in visited:
                    visited.add(name)
                    out.append(name)
                    dfs(name)

        dfs(step_name)
        return out

    def _execute_step(self, m: Dict[str, Any], step_def: StepDef) -> None:
        name = step_def.name
        entry = self._step_entry(m, name)

        # Resume logic
        if self.resume and not self.force and self._is_step_valid_for_resume(m, name, step_def):
            logger.info(f"Skipping {name} (Already Completed)")
            # If ingest resumed, also load context from manifest
            if name == "ingest":
                entry_result = entry.get("result", {})
                self.ctx.ingest = entry_result
            return

        # If rerunning, invalidate downstream deterministically
        self._invalidate_downstream(m, name)

        entry["status"] = "RUNNING"
        entry["started_at"] = now_iso()
        
        # Update parent manifest with current step and timestamp
        m["current_step"] = name
        m["updated_at"] = now_iso()
        self._save_manifest(m)

        logger.info(f"Running Step: {name}")
        t0 = time.time()
        
        # Heartbeat thread to keep updated_at fresh during long steps
        import threading
        heartbeat_stop = threading.Event()
        
        def heartbeat_worker():
            while not heartbeat_stop.is_set():
                heartbeat_stop.wait(3.0)  # 3-second heartbeat
                if not heartbeat_stop.is_set():
                    m_live = self._load_manifest()
                    m_live["updated_at"] = now_iso()
                    self._save_manifest(m_live)
        
        heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
        heartbeat_thread.start()
        
        try:
            # Failure Injection (Test Hook)
            fail_target = os.environ.get("SESSION_FAIL_STEP")
            if fail_target == step_def.name:
                time.sleep(1) # Ensure started_at is distinct
                raise RuntimeError(f"Simulated failure for step: {step_def.name}")

            result = step_def.func(self.ctx)
            duration_ms = int((time.time() - t0) * 1000)

            # Update context if ingest
            if name == "ingest":
                self.ctx.ingest = result

            # Record artifacts and hashes
            artifacts: List[Dict[str, Any]] = []
            for p in step_def.artifact_paths(result):
                # Ensure path is recorded relative if possible, or absolute?
                # User used absolute in skeleton. Stick to absolute for now or normalized.
                artifacts.append({"path": str(p), "hash": sha256_file(p)})

            entry["status"] = "COMPLETED"
            entry["ended_at"] = now_iso()
            entry["duration_ms"] = duration_ms
            entry["artifacts"] = artifacts
            entry["result"] = result  # Keep for strict resume checks
            
            # Clear current_step on completion
            m["current_step"] = None
            m["updated_at"] = now_iso()
            self._save_manifest(m)
        except Exception as e:
            duration_ms = int((time.time() - t0) * 1000)
            entry["status"] = "FAILED"
            entry["ended_at"] = now_iso()
            entry["duration_ms"] = duration_ms
            entry["error"] = {"type": e.__class__.__name__, "message": str(e)}
            
            # Add failure fields to manifest
            m["status"] = "FAILED"
            m["error_step"] = name
            m["error_code"] = e.__class__.__name__
            m["error_message"] = str(e)[:200]  # Truncate to 200 chars
            m["current_step"] = None  # No longer running
            m["updated_at"] = now_iso()
            
            # Log path if artifacts exist
            artifacts_dir = self.session_dir / "artifacts" / name
            if artifacts_dir.exists():
                m["log_path"] = str(artifacts_dir.relative_to(self.session_dir))
            
            self._save_manifest(m)
            raise
        
        finally:
            # Stop heartbeat thread
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=5.0)

    def _topo_order(self) -> List[str]:
        # Kahn's algorithm or DFS
        # We need full order.
        adj = {name: set(s.deps) for name, s in self.steps.items()}
        order = []
        while True:
            # Find nodes with no deps
            ready = sorted([n for n, deps in adj.items() if not deps])
            if not ready:
                break
            for n in ready:
                order.append(n)
                del adj[n]
                # Remove n from others deps (conceptually, but deps list is static)
                # Actually we need to remove n from remaining items deps set
                for deps in adj.values():
                    deps.discard(n)
        
        if adj:
            # Cycle detected or unreachable?
            raise RuntimeError(f"Cycle or dependency error: {adj.keys()}")
            
        return order

    def run(self) -> Dict[str, Any]:
        self._init_dirs()
        self._register_steps()

        # Load or create manifest
        m = self._load_manifest()
        
        # Safety check: if recovering from RUNNING, mark failed first
        if m["status"] == "RUNNING":
             logger.warning("Resuming unclosed session. Marking previous run FAILED.")
             m["status"] = "FAILED"
             
        # Set to RUNNING and write IMMEDIATELY (before heavy work)
        # This is the critical invariant: manifest must exist with RUNNING status
        # as soon as the run directory exists
        m["status"] = "RUNNING"
        m["started_at"] = m["started_at"] or now_iso()
        m["current_step"] = None  # No step running yet
        m["updated_at"] = now_iso()
        self._save_manifest(m)

        t0 = time.time()
        run_failed = False
        try:
            # 0. Check for stale RUNNING steps (Crash Recovery)
            stale_detected = False
            for s_name, s_rec in m.get("steps", {}).items():
                if s_rec.get("status") == "RUNNING":
                    logger.warning(f"Found stale RUNNING step '{s_name}'. Marking FAILED (StaleRun).")
                    s_rec["status"] = "FAILED"
                    s_rec["error"] = {"type": "StaleRun", "message": "Previous run crashed or was interrupted."}
                    stale_detected = True
            
            if stale_detected:
                if m.get("status") == "RUNNING":
                     m["status"] = "FAILED"
                self._save_manifest(m)

            # 1. Ingest (always first, special handling for context)
            # Ensure ingest is part of order if needed, or handled via dependency.
            # We defined "ingest" as a step, so it will be in topo order.
            
            # Determine execution order
            full_order = self._topo_order()

            # Apply requested steps if provided
            if self.requested_steps:
                # If granular steps requested, we must still respect dependencies?
                # Usually we want to run dependencies too.
                # Strategy: Add all dependencies of requested steps.
                to_run = set(self.requested_steps)
                
                # Expand dependencies
                # Simple iterative expansion
                changed = True
                while changed:
                    changed = False
                    current_deps = set()
                    for s in to_run:
                        if s in self.steps:
                            current_deps.update(self.steps[s].deps)
                    
                    if not current_deps.issubset(to_run):
                        to_run.update(current_deps)
                        changed = True
                
                order = [s for s in full_order if s in to_run]
            else:
                order = full_order

            # Always finalize with a bundle step (best-effort packaging contract).
            # This runs even when a subset of steps is requested.
            if "bundle" not in order:
                order.append("bundle")

            logger.info(f"Starting Session {self.run_id} for {self.input_path.name}")
            logger.info(f"Execution Order: {order}")

            for step_name in order:
                if step_name not in self.steps:
                    logger.warning(f"Unknown step {step_name}, skipping")
                    continue
                
                step_def = self.steps[step_name]
                self._execute_step(m, step_def)

            m["status"] = "COMPLETED"
            m["ended_at"] = now_iso()
            m["duration_ms"] = int((time.time() - t0) * 1000)
            self._save_manifest(m)
            
        except Exception as e:
            run_failed = True
            logger.error(f"Session Failed: {e}", exc_info=True)
            m["status"] = "FAILED"
            m["warnings"].append(str(e))
            self._save_manifest(m)
            raise e
        finally:
            # Always try to produce a Meeting Pack manifest (and any available artifacts) as a value artifact.
            # On failure, this creates a partial bundle that marks missing artifacts as absent.
            try:
                from harness.meeting_pack import build_meeting_pack
                build_meeting_pack(self.session_dir)
            except Exception as e:
                logger.error(f"Failed to build Meeting Pack bundle: {e}")

            # Always export the legacy zip (partial on failure).
            self._export_partial_bundle(final=not run_failed)
            
        return m

    def _export_partial_bundle(self, final: bool = False):
        """Export session artifact bundle."""
        try:
            import zipfile
            
            bundle_name = "session_bundle.zip"
            bundle_path = self.ctx.bundle_dir / bundle_name
            
            with zipfile.ZipFile(bundle_path, 'w') as zf:
                # Manifest
                zf.write(self.manifest_path, "manifest.json")
                
                # Logs
                for log in self.ctx.logs_dir.glob("*"):
                    zf.write(log, f"logs/{log.name}")
                    
                # Artifacts
                # Iterate manifest steps
                m = self._load_manifest()
                for step_name, step_rec in m.get("steps", {}).items():
                    # We might export artifacts even if FAILED? Usually only COMPLETED.
                    if step_rec.get("status") == "COMPLETED":
                        for art in step_rec.get("artifacts", []):
                            p = Path(art["path"])
                            # p is absolute (from _execute_step)
                            if p.exists():
                                rel_path = p.relative_to(self.session_dir) if p.is_relative_to(self.session_dir) else f"artifacts/{p.name}"
                                zf.write(p, rel_path)
                                
            logger.info(f"Bundle {'(Final)' if final else '(Partial)'} created: {bundle_path}")
            
            from harness.runner_schema import compute_file_hash
            m = self._load_manifest()
            m["final"]["bundle_path"] = str(bundle_path.relative_to(self.session_dir))
            m["final"]["bundle_hash"] = compute_file_hash(bundle_path)
            self._save_manifest(m)
            
        except Exception as e:
            logger.error(f"Failed to export bundle: {e}")
