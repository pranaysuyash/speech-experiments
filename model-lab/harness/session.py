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
    
    # Progress callback
    on_progress: Optional[Callable[[], None]] = None

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


@dataclass
class ArtifactMetadata:
    """
    Semantic artifact metadata. Locked schema for Phase 2.
    
    - id: Primary key for download endpoint
    - filename: User-visible name
    - role: Semantic purpose (transcript, processed_audio, etc.)
    - produced_by: Step that created it
    - path: Internal path relative to session_dir (never exposed to client)
    - size_bytes: File size from stat()
    - content_type: MIME type
    - downloadable: Explicit boolean
    """
    id: str
    filename: str
    role: str
    produced_by: str
    path: str  # Relative to session_dir
    size_bytes: int
    content_type: str
    downloadable: bool = True

    def to_manifest_dict(self) -> Dict[str, Any]:
        """Full dict for manifest storage."""
        return {
            "id": self.id,
            "filename": self.filename,
            "role": self.role,
            "produced_by": self.produced_by,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "content_type": self.content_type,
            "downloadable": self.downloadable,
        }

    def to_api_dict(self) -> Dict[str, Any]:
        """Dict for API exposure (path and content_type hidden)."""
        return {
            "id": self.id,
            "filename": self.filename,
            "role": self.role,
            "produced_by": self.produced_by,
            "size_bytes": self.size_bytes,
            "downloadable": self.downloadable,
        }


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(tmp), str(path))


def compute_run_id(input_hash: str) -> str:
    # Deterministic enough: timestamp + short hash + uuid for uniqueness
    import uuid
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    unique = uuid.uuid4().hex[:6]  # Add randomness to prevent collisions
    short = hashlib.sha256((ts + input_hash + unique).encode("utf-8")).hexdigest()[:10]
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
        # Get input file size upfront
        input_size_bytes = 0
        try:
            input_size_bytes = self.input_path.stat().st_size
        except Exception:
            pass
        
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": self.run_id,
            "input": {
                "original_path": str(self.input_path),
                "input_hash": self.input_hash,
                "size_bytes": input_size_bytes,
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
            "failure_step": None,  # Authoritative: step that raised exception
        }

    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return self._default_manifest()

    # Fixed check names - stable contract, never change dynamically
    EVAL_CHECK_NAMES = [
        "bundle_manifest_present",
        "bundle_manifest_parseable",
        "transcript_present",
        "summary_present",
        "action_items_present",
        "decisions_present",
        "asr_output_present",
        "diarization_present",
        "alignment_present",
        "run_terminal_status_ok",
    ]

    def _write_eval_json(self, manifest: Dict[str, Any]) -> None:
        """Write eval.json contract for results/findings UI.

        Contract:
        - Written after the bundle attempt (best-effort)
        - Written at run root: <run_dir>/eval.json
        - Does not mutate the run manifest semantics (status/error fields)
        - MODE: MODEL_LAB_EVAL_MODE=enriched (default) | identity
        """
        import os
        eval_mode = os.environ.get("MODEL_LAB_EVAL_MODE", "enriched")
        if eval_mode == "enriched_v0": # Force new mode if strictly needed, but let's stick to "enriched" or just update default.
            # We want to be backward compatible in "enriched" mode?
            # User request implies extending default behavior.
            pass

        eval_path = self.session_dir / "eval.json"
        
        # Extended V1: Load run_request.json to get context (use_case_id)
        use_case_id = None
        model_id = None
        run_request_path = self.session_dir / "run_request.json"
        if run_request_path.exists():
            try:
                rr = json.loads(run_request_path.read_text(encoding="utf-8"))
                use_case_id = rr.get("use_case_id")
                model_id = rr.get("model_id")
            except Exception:
                pass
        
        from datetime import datetime
        
        if eval_mode == "identity":
            # Identity mode: schema header only, empty checks/findings
            run_id = manifest.get("run_id") or self.run_id
            eval_data = {
                "schema_version": "1",
                "run_id": run_id,
                "schema_version": "1",
                "run_id": run_id,
                "use_case_id": use_case_id, # Updated from None
                "model_id": model_id, # Updated from None
                "params": manifest.get("config", {}),
                "params": manifest.get("config", {}),
                "metrics": {},
                "checks": [],
                "findings": [],
                "generated_at": datetime.now().isoformat()
            }
            atomic_write_json(eval_path, eval_data)
            return
        
        # Enriched mode (default): 10 fixed checks based on filesystem + manifest
        checks = []
        findings = []
        
        # Check 1: bundle_manifest_present
        bundle_manifest_path = self.session_dir / "bundle" / "bundle_manifest.json"
        manifest_exists = bundle_manifest_path.exists()
        checks.append({
            "name": "bundle_manifest_present",
            "passed": manifest_exists,
            "severity": "info",
            "message": "Bundle manifest present" if manifest_exists else "Bundle manifest not found",
            "evidence_paths": ["bundle/bundle_manifest.json"] if manifest_exists else []
        })
        
        # Check 2: bundle_manifest_parseable
        manifest_parseable = False
        if manifest_exists:
            try:
                json.loads(bundle_manifest_path.read_text())
                manifest_parseable = True
            except Exception:
                pass
        checks.append({
            "name": "bundle_manifest_parseable",
            "passed": manifest_parseable,
            "severity": "warn" if manifest_exists and not manifest_parseable else "info",
            "message": "Bundle manifest valid JSON" if manifest_parseable else ("Bundle manifest invalid JSON" if manifest_exists else "N/A - no manifest"),
            "evidence_paths": ["bundle/bundle_manifest.json"] if manifest_exists else []
        })
        if manifest_exists and not manifest_parseable:
            findings.append({
                "finding_id": "bundle:manifest_invalid",
                "severity": "medium",
                "category": "system",
                "title": "Bundle manifest is not valid JSON",
                "details": "bundle_manifest.json exists but cannot be parsed",
                "evidence_paths": ["bundle/bundle_manifest.json"]
            })
        
        # Checks 3-6: artifact presence
        artifacts = [
            ("transcript_present", "bundle/transcript.txt", "Transcript"),
            ("summary_present", "bundle/summary.md", "Summary"),
            ("action_items_present", "bundle/action_items.csv", "Action items"),
            ("decisions_present", "bundle/decisions.md", "Decisions"),
        ]
        for check_name, rel_path, label in artifacts:
            exists = (self.session_dir / rel_path).exists()
            checks.append({
                "name": check_name,
                "passed": exists,
                "severity": "info",
                "message": f"{label} generated" if exists else f"{label} not generated",
                "evidence_paths": [rel_path] if exists else []
            })
        
        # Checks 7-9: processing outputs
        processing = [
            ("asr_output_present", "artifacts/asr.json", "ASR output"),
            ("diarization_present", "artifacts/diarization.json", "Diarization output"),
            ("alignment_present", "artifacts/alignment.json", "Alignment output"),
        ]
        for check_name, rel_path, label in processing:
            exists = (self.session_dir / rel_path).exists()
            checks.append({
                "name": check_name,
                "passed": exists,
                "severity": "info",
                "message": f"{label} present" if exists else f"{label} not present",
                "evidence_paths": [rel_path] if exists else []
            })
        
        # Check 10: run_terminal_status_ok
        status = manifest["status"]
        status_ok = status == "COMPLETED"
        checks.append({
            "name": "run_terminal_status_ok",
            "passed": status_ok,
            "severity": "fail" if status in ("FAILED", "STALE") else "info",
            "message": f"Run completed successfully" if status_ok else f"Run status: {status}",
            "evidence_paths": ["manifest.json"]
        })
        
        # Generate finding for failures
        if status == "FAILED":
            error_step = manifest.get("error_step", "unknown")
            error_msg = manifest.get("error_message", "Unknown error")
            findings.append({
                "finding_id": f"{manifest.get('error_code', 'Error')}:{error_step}",
                "severity": "high",
                "category": "system",
                "title": f"Run failed at {error_step}",
                "details": error_msg[:200],
                "evidence_paths": ["manifest.json"]
            })
        elif status == "STALE":
            findings.append({
                "finding_id": "system:stale_run",
                "severity": "medium",
                "category": "system",
                "title": "Run stopped responding",
                "details": f"No heartbeat since {manifest.get('updated_at', 'unknown')}",
                "evidence_paths": ["manifest.json"]
            })
        
        score_cards = []
        if use_case_id == "meeting_smoke":
             score_cards = self._compute_proxy_scores()

        run_id = manifest.get("run_id") or self.run_id
        eval_data = {
            "schema_version": "1",
            "run_id": run_id,
            "use_case_id": use_case_id,
            "model_id": model_id,
            "params": manifest.get("config", {}),
            "metrics": {},  # V1: no computed metrics
            "score_cards": score_cards,
            "checks": checks,
            "findings": findings,
            "generated_at": datetime.now().isoformat()
        }
        
        atomic_write_json(eval_path, eval_data)

    def _compute_proxy_scores(self) -> List[Dict[str, Any]]:
        """Compute deterministic proxy scores for meeting_smoke."""
        scores = []
        
        # 1. Artifact Completeness
        # Check for artifacts (try .json first for canonical, then .txt legacy)
        required_checks = [
            (["bundle/transcript.json", "bundle/transcript.txt"], "transcript"),
            (["bundle/summary.md"], "summary"),
            (["bundle/action_items.csv"], "action_items"),
        ]
        found = []
        for paths, _ in required_checks:
            if any((self.session_dir / p).exists() for p in paths):
                found.append(next(p for p in paths if (self.session_dir / p).exists()))
        
        score_completeness = int((len(found) / len(required_checks)) * 100)
        scores.append({
            "name": "artifact_completeness",
            "label": "Artifact Completeness",
            "type": "proxy",
            "score": score_completeness,
            "evidence_paths": found,
            "notes": f"Found {len(found)}/{len(required_checks)} artifacts"
        })

        # 2. Transcript Length OK
        # Try .json first (canonical), then .txt (legacy)
        t_path = None
        for candidate in ["bundle/transcript.json", "bundle/transcript.txt"]:
            p = self.session_dir / candidate
            if p.exists():
                t_path = p
                break
                
        t_ok = False
        t_len = 0
        t_text = None
        if t_path:
            try:
                if t_path.suffix == ".json":
                    # Parse Meeting Pack canonical format
                    data = json.loads(t_path.read_text(encoding="utf-8"))
                    # Extract text from segments
                    if "segments" in data and isinstance(data["segments"], list):
                        t_text = " ".join(seg.get("text", "") for seg in data["segments"] if isinstance(seg, dict))
                        t_len = len(t_text)
                    elif "text" in data:
                        t_text = data["text"]
                        t_len = len(t_text)
                    else:
                        t_len = 0
                else:
                    # Plain text file
                    t_text = t_path.read_text(encoding="utf-8")
                    t_len = len(t_text)
                    
                if t_len > 100:
                    t_ok = True
            except:
                pass
                
        scores.append({
            "name": "transcript_length_ok",
            "label": "Transcript Viability",
            "type": "proxy",
            "score": 100 if t_ok else 0,
            "evidence_paths": [str(t_path.relative_to(self.session_dir))] if t_path else [],
            "notes": f"Length: {t_len} chars" if t_path else "Missing"
        })

        # 3. Action Items Parseable
        ai_path = self.session_dir / "bundle" / "action_items.csv"
        ai_ok = False
        if ai_path.exists():
             try:
                 import csv
                 with ai_path.open("r", encoding="utf-8") as f:
                     reader = csv.reader(f)
                     rows = list(reader)
                     if len(rows) > 0: # Has header at least
                          ai_ok = True
             except:
                 pass
        scores.append({
            "name": "action_items_parseable",
            "label": "Action Items Format",
            "type": "proxy",
            "score": 100 if ai_ok else 0,
            "evidence_paths": ["bundle/action_items.csv"] if ai_path.exists() else [],
            "notes": "Valid CSV" if ai_ok else "Invalid/Missing"
        })

        # 4. Summary Non-Empty
        s_path = self.session_dir / "bundle" / "summary.md"
        s_ok = False
        if s_path.exists():
            if s_path.stat().st_size > 50:
                 s_ok = True
        scores.append({
             "name": "summary_nonempty",
             "label": "Summary Content",
             "type": "proxy",
             "score": 100 if s_ok else 0,
             "evidence_paths": ["bundle/summary.md"] if s_path.exists() else [],
             "notes": "Content > 50 bytes" if s_ok else "Empty/Missing"
        })

        return scores

    def _save_manifest(self, m: Dict[str, Any]) -> None:
        atomic_write_json(self.manifest_path, m)

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
             return run_asr(ctx.audio_path, ctx.artifacts_dir, config=self.extra_config.get("asr", {}), progress_callback=ctx.on_progress)

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
            try:
                from harness.diarization import run_diarization
                diar_config = self.extra_config.get("diarization", {})
                model_name = diar_config.get("model_name", "heuristic_diarization")
                output_dir = ctx.artifacts_dir / "diarization"
                output_dir.mkdir(parents=True, exist_ok=True)
                artifact_path = run_diarization(ctx.audio_path, model_name, output_dir)
                return {"artifacts": [{"path": str(artifact_path)}]}
            except Exception as e:
                logger.warning(f"Diarization failed (non-fatal): {e}")
                return {"artifacts": [], "warning": str(e)}
            
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
        
        # CRITICAL: Resolve and persist config BEFORE execution
        # If step fails/hangs, this data is still available for debugging
        if name == "asr":
            from harness.asr import resolve_asr_config
            asr_user_config = self.extra_config.get("asr", {})
            resolved = resolve_asr_config(asr_user_config)
            entry["resolved_config"] = resolved.to_dict()
            logger.info(f"ASR resolved: {resolved.model_id} on {resolved.device}")
        
        # Update parent manifest with current step and timestamp
        m["current_step"] = name
        m["updated_at"] = now_iso()
        self._save_manifest(m)  # MANDATORY FLUSH - resolved_config now persisted

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
                    
                    # Update semantic progress if available
                    if progress_state["last_semantic_progress_at"]:
                        step_entry = m_live["steps"].get(name)
                        if step_entry:
                            step_entry["last_semantic_progress_at"] = progress_state["last_semantic_progress_at"]
                            
                    self._save_manifest(m_live)
        
        # Shared state for progress (thread-safe by GIL mostly, but simple dict is fine)
        progress_state = {"last_semantic_progress_at": None}
        
        def on_progress():
            progress_state["last_semantic_progress_at"] = now_iso()
            
        self.ctx.on_progress = on_progress
        
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

            # Record artifacts with semantic schema (Phase 2)
            artifacts: List[Dict[str, Any]] = []
            
            if name == "asr":
                # Use new semantic artifact schema for ASR
                for p in step_def.artifact_paths(result):
                    path_obj = Path(p)
                    if path_obj.exists():
                        # Compute relative path from session_dir
                        try:
                            rel_path = str(path_obj.relative_to(self.session_dir))
                        except ValueError:
                            rel_path = str(path_obj)  # Fallback to absolute if outside
                        
                        artifact = ArtifactMetadata(
                            id="asr_transcript",
                            filename=path_obj.name,
                            role="transcript",
                            produced_by="asr",
                            path=rel_path,
                            size_bytes=path_obj.stat().st_size,
                            content_type="application/json",
                            downloadable=True
                        )
                        artifacts.append(artifact.to_manifest_dict())
            else:
                # Legacy format for other steps (to be migrated later)
                for p in step_def.artifact_paths(result):
                    artifacts.append({"path": str(p), "hash": sha256_file(p)})

            entry["status"] = "COMPLETED"
            entry["ended_at"] = now_iso()
            entry["duration_ms"] = duration_ms
            entry["artifacts"] = artifacts
            entry["result"] = result  # Keep for strict resume checks

            
            # Persist final semantic progress timestamp
            if progress_state["last_semantic_progress_at"]:
                entry["last_semantic_progress_at"] = progress_state["last_semantic_progress_at"]
            
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
            m["failure_step"] = name  # AUTHORITATIVE: Captured at exception site
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
            # Non-pipeline failure: explicitly set failure_step to None
            if "failure_step" not in m or m["failure_step"] is None:
                m["failure_step"] = None  # Pre-pipeline or setup failure
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

            # Write eval.json after bundle attempt (success or failure).
            try:
                self._write_eval_json(m)
            except Exception as e:
                logger.error(f"Failed to write eval.json: {e}")

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
