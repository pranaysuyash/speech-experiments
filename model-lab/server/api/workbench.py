from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from harness.session import SessionRunner
from server.api.pipelines import build_pipeline_config
from server.services.lifecycle import (
    RunnerBusyError,
    launch_run_worker,
    release_worker,
    try_acquire_worker,
)

router = APIRouter(prefix="/api/workbench", tags=["workbench"])


def _runs_root() -> Path:
    return Path(os.environ.get("MODEL_LAB_RUNS_ROOT", "runs")).resolve()


def _inputs_root() -> Path:
    return Path(os.environ.get("MODEL_LAB_INPUTS_ROOT", "inputs")).resolve()


def _safe_filename(name: str) -> str:
    # Keep this conservative; storage is for provenance, not pretty names.
    name = (name or "").strip()
    if not name:
        return "upload"
    name = name.replace("\\", "_").replace("/", "_")
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)[:120] or "upload"


def _parse_csv(value: str | None) -> list[str]:
    """Parse a comma-separated string into a list of non-empty, stripped items."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge two dictionaries.

    Override values take precedence. Nested dicts are merged recursively.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# Preset registry - source of truth for available step presets
PRESETS = {
    "ingest": {
        "label": "Ingest Only",
        "description": "Fast preprocessing: normalize audio and create bundle structure",
        "steps": ["ingest"],
    },
    "fast_asr_only": {
        "label": "Fast ASR",
        "description": "Quick transcription without diarization or summarization",
        "steps": ["ingest", "asr"],
    },
    "asr_with_diarization": {
        "label": "ASR + Diarization",
        "description": "Transcription with speaker identification",
        "steps": ["ingest", "asr", "diarization", "alignment"],
    },
    "diarization_focus": {
        "label": "Diarization Focus",
        "description": "Speaker analysis without LLM-based summarization",
        "steps": ["ingest", "asr", "diarization", "alignment"],
    },
    "full": {
        "label": "Full Pipeline",
        "description": "Complete pipeline with transcription, diarization, summarization and action items",
        "steps": None,  # None means all steps
    },
}


@router.get("/presets")
def get_presets() -> list[dict]:
    """
    Return available step presets.

    This is the source of truth for what presets experiments can use.
    """
    return [
        {"steps_preset": key, "label": meta["label"], "description": meta.get("description")}
        for key, meta in PRESETS.items()
    ]


@router.get("/steps")
def get_available_steps() -> list[dict]:
    """Return available pipeline steps with descriptions."""
    return [
        {"name": "ingest", "deps": [], "description": "Audio normalization and preprocessing"},
        {"name": "asr", "deps": ["ingest"], "description": "Speech-to-text transcription"},
        {"name": "diarization", "deps": ["ingest"], "description": "Speaker identification"},
        {
            "name": "alignment",
            "deps": ["asr", "diarization"],
            "description": "Merge ASR with speaker labels",
        },
        {"name": "chapters", "deps": ["alignment"], "description": "Topic segmentation"},
        {
            "name": "summarize_by_speaker",
            "deps": ["alignment"],
            "description": "Per-speaker summary (LLM)",
        },
        {
            "name": "action_items_assignee",
            "deps": ["alignment"],
            "description": "Extract action items (LLM)",
        },
        {
            "name": "bundle",
            "deps": [
                "ingest",
                "asr",
                "diarization",
                "alignment",
                "chapters",
                "summarize_by_speaker",
                "action_items_assignee",
            ],
            "description": "Package as Meeting Pack",
        },
    ]


def start_run_from_path(
    input_path: Path,
    use_case_id: str,
    steps_preset: str,
    *,
    experiment_id: str | None = None,
    candidate_id: str | None = None,
    source_ref: dict[str, Any] | None = None,
    candidate_snapshot: dict[str, Any] | None = None,
    config_overrides: dict[str, Any] | None = None,
    pipeline_config: dict[str, Any] | None = None,  # Custom pipeline configuration
) -> dict[str, Any]:
    """
    Start a run from a file path (not multipart upload).

    This is the pure function callable by experiments API.

    Args:
        input_path: Path to input media file
        use_case_id: Use case identifier
        steps_preset: Preset name (must exist in PRESETS)
        experiment_id: Optional experiment this run belongs to
        candidate_id: Optional candidate ID within experiment
        source_ref: Optional source reference metadata
        config_overrides: Optional configuration overrides (e.g. device_preference)
        pipeline_config: Optional custom pipeline configuration with resolved_steps

    Returns:
        {run_id, run_dir, console_url}

    Raises:
        RunnerBusyError: If runner is already processing another job
        HTTPException: If preset is invalid
    """
    if not try_acquire_worker():
        raise RunnerBusyError("Runner is busy with another job")

    try:
        # Determine steps: pipeline_config takes precedence over preset
        if pipeline_config and pipeline_config.get("resolved_steps"):
            steps = pipeline_config["resolved_steps"]
        elif steps_preset not in PRESETS:
            raise HTTPException(
                status_code=400, detail=f"Invalid steps_preset. Available: {list(PRESETS.keys())}"
            )
        else:
            steps = PRESETS[steps_preset]["steps"]

        # Merge config: candidate params (base) + config overrides (user)
        # Candidate params come from the candidate definition (e.g., model_type, model_name)
        # Config overrides come from the UI (e.g., device_preference)
        run_config = {}
        if candidate_snapshot and candidate_snapshot.get("params"):
            run_config = _deep_merge(run_config, candidate_snapshot["params"])
        if config_overrides:
            run_config = _deep_merge(run_config, config_overrides)

        # Build preprocessing config if provided in pipeline_config
        ingest_config = None
        if pipeline_config and pipeline_config.get("config"):
            from harness.pipeline_config import PipelineConfig

            try:
                cfg = PipelineConfig.from_dict(pipeline_config["config"])
                ingest_config = cfg.to_ingest_config()
            except Exception:
                pass  # Fall back to no preprocessing

        runner = SessionRunner(
            input_path, _runs_root(), steps=steps, config=run_config, preprocessing=ingest_config
        )

        now = datetime.now(UTC)

        # Compute file info for run_request.json
        file_bytes = input_path.stat().st_size
        import hashlib

        sha256_hex = hashlib.sha256(input_path.read_bytes()).hexdigest()

        # Construct run_request data
        run_request_data = {
            "schema_version": "1",
            "requested_at": now.isoformat(),
            "source": "experiment" if experiment_id else "workbench",
            "use_case_id": use_case_id,
            "steps_preset": steps_preset,
            "steps_requested": steps,
            "filename_original": input_path.name,
            "content_type": None,
            "bytes_uploaded": file_bytes,
            "sha256": sha256_hex,
            "config": run_config,
        }

        if experiment_id:
            run_request_data["experiment_id"] = experiment_id
            run_request_data["candidate_id"] = candidate_id
            run_request_data["source_ref"] = source_ref
            run_request_data["candidate_snapshot"] = candidate_snapshot

        # Add pipeline configuration if custom pipeline was used
        if pipeline_config:
            run_request_data["pipeline_config"] = pipeline_config

        # Launch worker
        result = launch_run_worker(runner, run_request_data, background=True)

        return {
            "run_id": runner.run_id,
            "input_hash": sha256_hex,
            "run_dir": str(runner.session_dir),
            "console_url": f"/runs/{runner.run_id}",
            "worker_pid": result.get("worker_pid"),
        }
    except Exception:
        release_worker()
        raise


async def _save_upload_to_disk(upload: UploadFile, dest: Path, max_bytes: int) -> tuple[int, str]:
    """
    Save upload to disk while computing sha256.

    Returns (bytes_written, sha256_hex).
    """
    import hashlib

    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    h = hashlib.sha256()

    with dest.open("wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(status_code=413, detail="Upload too large")
            h.update(chunk)
            f.write(chunk)

    return total, h.hexdigest()


@router.post("/runs")
async def create_workbench_run(
    file: UploadFile = File(...),
    use_case_id: str = Form(...),
    steps_preset: str = Form("full"),
    config: str | None = Form(None),
    steps: str | None = Form(None),
    preprocessing: str | None = Form(None),
    pipeline_template: str | None = Form(None),
) -> JSONResponse:
    """
    Create a run from the UI (multipart upload) and start processing asynchronously.

    Contract:
    - Returns quickly once run dir exists and manifest is RUNNING.
    - Single-worker guardrail: returns 409 RUNNER_BUSY if a run is already active.
    - Writes run_request.json at run root for reproducibility.
    - Accepts optional 'config' JSON string (e.g. {"device_preference": ["mps", "cpu"]}).

    Dynamic Pipeline Selection:
    - steps: Comma-separated list of steps (e.g., "ingest,asr,diarization")
    - preprocessing: Comma-separated preprocessing ops (e.g., "trim_silence,normalize_loudness")
    - pipeline_template: Name of a built-in template (overrides steps_preset)

    Priority: steps > pipeline_template > steps_preset
    """
    if not try_acquire_worker():
        return JSONResponse(status_code=409, content={"error_code": "RUNNER_BUSY"})

    try:
        max_upload = int(
            os.environ.get("MODEL_LAB_WORKBENCH_MAX_UPLOAD_BYTES", str(200 * 1024 * 1024))
        )
        now = datetime.now(UTC)
        yyyy_mm = now.strftime("%Y-%m")
        inputs_dir = _inputs_root() / "workbench" / yyyy_mm
        filename = _safe_filename(file.filename or "upload")
        dest = inputs_dir / f"{uuid4().hex}_{filename}"

        # Save upload and compute sha256
        bytes_uploaded, sha256_hex = await _save_upload_to_disk(file, dest, max_upload)

        if steps_preset not in PRESETS:
            raise HTTPException(
                status_code=400, detail=f"Invalid steps_preset. Available: {list(PRESETS.keys())}"
            )

        preset_steps = PRESETS[steps_preset]["steps"]

        # Parse config overrides
        config_overrides = {}
        if config:
            try:
                config_overrides = json.loads(config)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid config JSON")

        # Dynamic pipeline selection: steps > pipeline_template > steps_preset
        custom_steps = _parse_csv(steps)
        preprocessing_ops = _parse_csv(preprocessing)
        pipeline_template_name = pipeline_template.strip() if pipeline_template else None

        resolved_steps = preset_steps
        pipeline_config_payload = None

        if custom_steps or pipeline_template_name or preprocessing_ops:
            try:
                pipeline_cfg = build_pipeline_config(
                    template=pipeline_template_name,
                    steps=custom_steps or None,
                    preprocessing=preprocessing_ops or None,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

            resolved_steps = pipeline_cfg.resolve_dependencies()
            pipeline_config_payload = pipeline_cfg.to_dict()

        steps_for_runner = resolved_steps

        # Convert preprocessing ops to IngestConfig
        ingest_config = pipeline_cfg.to_ingest_config() if pipeline_config_payload else None

        runner = SessionRunner(
            dest,
            _runs_root(),
            steps=steps_for_runner,
            config=config_overrides,
            preprocessing=ingest_config,
        )
        # Construct run_request data
        run_request_data = {
            "schema_version": "1",
            "requested_at": now.isoformat(),
            "source": "workbench",
            "use_case_id": use_case_id,
            "steps_preset": steps_preset,
            "steps_requested": steps_for_runner,
            "filename_original": file.filename or "unknown",
            "content_type": file.content_type,
            "bytes_uploaded": bytes_uploaded,
            "sha256": sha256_hex,
            "config": config_overrides,
        }

        # Attach dynamic pipeline details for reproducibility
        if pipeline_config_payload is not None:
            run_request_data["pipeline_template"] = pipeline_template_name
            run_request_data["steps_custom"] = custom_steps or None
            run_request_data["preprocessing"] = preprocessing_ops or None
            run_request_data["pipeline_config"] = pipeline_config_payload

        # Launch worker
        launch_run_worker(runner, run_request_data, background=True)

        # Write UI metadata without mutating the run manifest (multi-agent safe).
        try:
            meta: dict[str, Any] = {
                "use_case_id": use_case_id,
                "steps_preset": steps_preset,
                "input_rel_path": str(dest.relative_to(_inputs_root()))
                if dest.is_relative_to(_inputs_root())
                else str(dest),
                "created_at": now.isoformat(),
            }
            (runner.session_dir / "workbench.json").write_text(
                json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
            )
        except Exception:
            # Best-effort only.
            pass

        return JSONResponse(
            status_code=200,
            content={
                "run_id": runner.run_id,
                "run_dir": str(runner.session_dir),
                "console_url": f"/runs/{runner.run_id}",
            },
        )
    except Exception:
        release_worker()
        raise
