"""
Pipelines API - Dynamic step selection and pipeline management.

Provides endpoints for:
- Listing available steps and preprocessing operators
- Listing and retrieving pipeline templates
- Creating custom pipeline configurations
- Running ad-hoc pipelines with user-selected steps
- User-defined pipeline templates (CRUD)
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from harness.pipeline_config import (
    PIPELINE_TEMPLATES,
    PREPROCESSING_REGISTRY,
    STEP_REGISTRY,
    PipelineConfig,
    get_pipeline_template,
    list_available_steps,
    list_pipeline_templates,
    list_preprocessing_ops,
    validate_pipeline_config,
)

router = APIRouter(prefix="/api/pipelines", tags=["pipelines"])


# ============================================================================
# User Templates Storage
# ============================================================================


def _user_templates_path() -> Path:
    return Path(os.environ.get("MODEL_LAB_DATA_ROOT", "data")).resolve() / "user_templates.json"


def _load_user_templates() -> dict[str, dict[str, Any]]:
    path = _user_templates_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_user_templates(templates: dict[str, dict[str, Any]]) -> None:
    path = _user_templates_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(templates, indent=2, sort_keys=True), encoding="utf-8")


# ============================================================================
# Request/Response Models
# ============================================================================


class PipelineRunRequest(BaseModel):
    """Request body for creating a dynamic pipeline run."""

    # Either use a template name OR provide custom steps
    template: str | None = None

    # Custom step selection (if not using template)
    steps: list[str] | None = None

    # Preprocessing operators to apply
    preprocessing: list[str] | None = None

    # Per-step configuration overrides
    config: dict[str, dict[str, Any]] | None = None

    # Device preference
    device_preference: list[str] | None = None

    # Run metadata
    name: str | None = None
    use_case_id: str | None = None


class PipelineValidateRequest(BaseModel):
    """Request body for validating a pipeline configuration."""

    steps: list[str]
    preprocessing: list[str] | None = None
    config: dict[str, dict[str, Any]] | None = None


class UserTemplateRequest(BaseModel):
    """Request body for creating/updating a user template."""

    name: str
    steps: list[str]
    preprocessing: list[str] | None = None
    description: str | None = None


class WorkflowSuggestion(BaseModel):
    """A task-aware workflow suggestion."""

    id: str
    label: str
    rationale: str
    template: str | None = None
    steps: list[str] | None = None
    preprocessing: list[str] = []
    estimated_profile: str = "balanced"


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/steps")
def get_available_steps() -> JSONResponse:
    """
    List all available pipeline steps with their dependencies and metadata.

    Returns:
        List of step definitions including:
        - name: Step identifier
        - deps: Required dependencies (must run before this step)
        - description: Human-readable description
        - produces: Artifact types this step produces
        - config_schema: Configuration options for this step
    """
    return JSONResponse(content=list_available_steps())


@router.get("/preprocessing")
def get_preprocessing_operators() -> JSONResponse:
    """
    List all available preprocessing operators.

    Returns:
        List of operator definitions including:
        - name: Operator identifier
        - description: What the operator does
        - params: Configurable parameters with defaults
    """
    return JSONResponse(content=list_preprocessing_ops())


@router.get("/templates")
def get_pipeline_templates() -> JSONResponse:
    """
    List all built-in pipeline templates.

    Templates are pre-configured pipelines for common use cases.
    """
    return JSONResponse(content=list_pipeline_templates())


def _build_workflow_suggestions(
    *,
    use_case_id: str | None = None,
    goal: str | None = None,
    realtime: bool = False,
    quality: str = "balanced",
) -> list[dict[str, Any]]:
    """
    Build deterministic workflow suggestions based on use-case intent.

    Notes:
    - All steps are optional at run time; these are suggested defaults.
    - Include an explicit full pipeline option for comprehensive analysis.
    """
    uc = (use_case_id or "").lower()
    task_goal = (goal or "").lower().strip()
    q = quality.lower().strip() or "balanced"

    # Inference from use_case_id if goal not explicit
    if not task_goal:
        if "diar" in uc or "speaker" in uc:
            task_goal = "diarization"
        elif "summary" in uc or "meeting" in uc:
            task_goal = "meeting"
        elif "asr" in uc or "transcrib" in uc:
            task_goal = "asr"
        else:
            task_goal = "general"

    suggestions: list[WorkflowSuggestion] = []

    # Always include a full pipeline recommendation
    suggestions.append(
        WorkflowSuggestion(
            id="full_pipeline",
            label="Full Pipeline",
            rationale="Complete end-to-end run with transcription, diarization, alignment, chaptering, summaries, actions, and bundle.",
            template="full_meeting",
            preprocessing=["trim_silence", "normalize_loudness"],
            estimated_profile="high_quality",
        )
    )

    if task_goal in {"asr", "general"}:
        if realtime:
            suggestions.append(
                WorkflowSuggestion(
                    id="asr_realtime",
                    label="ASR Realtime",
                    rationale="Low-latency transcription path; minimal steps and conservative preprocessing.",
                    template="fast_asr",
                    preprocessing=["trim_silence"],
                    estimated_profile="fast",
                )
            )
        else:
            suggestions.append(
                WorkflowSuggestion(
                    id="asr_fast",
                    label="ASR Fast",
                    rationale="Fast transcription without speaker pipeline.",
                    template="fast_asr",
                    preprocessing=["trim_silence"],
                    estimated_profile="fast",
                )
            )
            suggestions.append(
                WorkflowSuggestion(
                    id="asr_hq",
                    label="ASR High Quality",
                    rationale="ASR with stronger preprocessing; best for clean benchmarking and quality scoring.",
                    steps=["ingest", "asr", "bundle"],
                    preprocessing=["trim_silence", "normalize_loudness"],
                    estimated_profile="high_quality",
                )
            )

    if task_goal in {"diarization", "meeting"}:
        suggestions.append(
            WorkflowSuggestion(
                id="speaker_focus",
                label="Speaker Focus",
                rationale="Optimized for speaker segmentation + aligned transcript without full summarization overhead.",
                template="asr_diarization",
                preprocessing=["trim_silence", "normalize_loudness"],
                estimated_profile="balanced",
            )
        )

    if task_goal == "meeting":
        suggestions.append(
            WorkflowSuggestion(
                id="quick_summary",
                label="Quick Summary",
                rationale="Get transcript and summary quickly with fewer steps.",
                template="quick_summary",
                preprocessing=["trim_silence"],
                estimated_profile="balanced",
            )
        )

    # Quality override nudges preprocessing and ordering preference
    if q == "fast":
        for s in suggestions:
            if s.preprocessing and "normalize_loudness" in s.preprocessing:
                s.preprocessing = [op for op in s.preprocessing if op != "normalize_loudness"]
        suggestions.sort(key=lambda s: 0 if s.estimated_profile == "fast" else 1)
    elif q in {"high", "hq", "quality"}:
        suggestions.sort(key=lambda s: 0 if s.estimated_profile == "high_quality" else 1)

    return [s.model_dump() for s in suggestions]


@router.get("/suggestions")
def get_workflow_suggestions(
    use_case_id: str | None = None,
    goal: str | None = None,
    realtime: bool = False,
    quality: str = "balanced",
) -> JSONResponse:
    """
    Return task-aware suggested workflows.

    Query params:
    - use_case_id: current use case id (optional)
    - goal: one of asr/diarization/meeting/general (optional, inferred from use_case_id if omitted)
    - realtime: prioritize low-latency lanes
    - quality: fast|balanced|high
    """
    return JSONResponse(
        content=_build_workflow_suggestions(
            use_case_id=use_case_id,
            goal=goal,
            realtime=realtime,
            quality=quality,
        )
    )


@router.get("/templates/{name}")
def get_pipeline_template_detail(name: str) -> JSONResponse:
    """
    Get details of a specific pipeline template.
    """
    template = get_pipeline_template(name)
    if not template:
        raise HTTPException(
            status_code=404,
            detail=f"Template '{name}' not found. Available: {list(PIPELINE_TEMPLATES.keys())}",
        )

    return JSONResponse(
        content={
            **template.to_dict(),
            "resolved_steps": template.resolve_dependencies(),
        }
    )


@router.post("/validate")
def validate_pipeline(request: PipelineValidateRequest) -> JSONResponse:
    """
    Validate a pipeline configuration without running it.

    Returns:
        - valid: Boolean indicating if config is valid
        - errors: List of validation errors (if any)
        - resolved_steps: Steps in execution order with dependencies resolved
    """
    errors = validate_pipeline_config(request.model_dump())

    if errors:
        return JSONResponse(
            content={
                "valid": False,
                "errors": errors,
                "resolved_steps": [],
            }
        )

    try:
        config = PipelineConfig(
            steps=request.steps,
            preprocessing=request.preprocessing or [],
            config=request.config or {},
        )
        resolved = config.resolve_dependencies()

        return JSONResponse(
            content={
                "valid": True,
                "errors": [],
                "resolved_steps": resolved,
            }
        )
    except ValueError as e:
        return JSONResponse(
            content={
                "valid": False,
                "errors": [str(e)],
                "resolved_steps": [],
            }
        )


@router.post("/resolve")
def resolve_dependencies(request: PipelineValidateRequest) -> JSONResponse:
    """
    Resolve step dependencies and return execution order.

    Given a list of steps, returns the complete list including
    all required dependencies in correct execution order.
    """
    try:
        config = PipelineConfig(
            steps=request.steps,
            preprocessing=request.preprocessing or [],
        )
        resolved = config.resolve_dependencies()

        return JSONResponse(
            content={
                "requested_steps": request.steps,
                "resolved_steps": resolved,
                "added_dependencies": [s for s in resolved if s not in request.steps],
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/step/{name}")
def get_step_info(name: str) -> JSONResponse:
    """
    Get detailed information about a specific step.
    """
    if name not in STEP_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Step '{name}' not found. Available: {list(STEP_REGISTRY.keys())}",
        )

    info = STEP_REGISTRY[name]

    # Find which steps depend on this one
    dependents = [s for s, i in STEP_REGISTRY.items() if name in i.get("deps", [])]

    return JSONResponse(
        content={
            "name": name,
            "deps": info.get("deps", []),
            "dependents": dependents,
            "description": info.get("description", ""),
            "produces": info.get("produces", []),
            "config_schema": info.get("config_schema"),
            "required": info.get("required", False),
        }
    )


@router.get("/preprocessing/{name}")
def get_preprocessing_info(name: str) -> JSONResponse:
    """
    Get detailed information about a preprocessing operator.
    """
    if name not in PREPROCESSING_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Operator '{name}' not found. Available: {list(PREPROCESSING_REGISTRY.keys())}",
        )

    info = PREPROCESSING_REGISTRY[name]

    return JSONResponse(
        content={
            "name": name,
            "description": info.get("description", ""),
            "params": info.get("params", {}),
        }
    )


# ============================================================================
# Helper for creating pipeline runs (used by workbench endpoint)
# ============================================================================


def build_pipeline_config(
    template: str | None = None,
    steps: list[str] | None = None,
    preprocessing: list[str] | None = None,
    config: dict[str, dict[str, Any]] | None = None,
    device_preference: list[str] | None = None,
) -> PipelineConfig:
    """
    Build a PipelineConfig from request parameters.

    Priority:
    1. If template is provided, use it as base
    2. Override with provided steps/preprocessing/config
    """
    if template:
        base = get_pipeline_template(template)
        if not base:
            raise ValueError(f"Unknown template: {template}")

        # Create a copy and override with provided values
        return PipelineConfig(
            name=template,
            description=base.description,
            steps=steps if steps else base.steps,
            preprocessing=preprocessing if preprocessing else base.preprocessing,
            config={**base.config, **(config or {})},
            device_preference=device_preference or base.device_preference,
        )

    # Custom pipeline
    return PipelineConfig(
        name="custom",
        description="User-defined pipeline",
        steps=steps or ["ingest", "asr"],
        preprocessing=preprocessing or [],
        config=config or {},
        device_preference=device_preference or ["mps", "cuda", "cpu"],
    )


# ============================================================================
# User Templates Endpoints
# ============================================================================


@router.get("/user-templates")
def list_user_templates() -> JSONResponse:
    """
    List all user-defined pipeline templates.
    """
    templates = _load_user_templates()
    return JSONResponse(
        content=[
            {
                "name": name,
                "steps": data.get("steps", []),
                "preprocessing": data.get("preprocessing", []),
                "description": data.get("description", ""),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
            }
            for name, data in templates.items()
        ]
    )


@router.post("/user-templates")
def create_user_template(request: UserTemplateRequest) -> JSONResponse:
    """
    Create or update a user-defined pipeline template.
    """
    # Validate steps
    errors = validate_pipeline_config(
        {
            "steps": request.steps,
            "preprocessing": request.preprocessing or [],
        }
    )
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    templates = _load_user_templates()
    now = datetime.now(UTC).isoformat()

    is_update = request.name in templates

    templates[request.name] = {
        "steps": request.steps,
        "preprocessing": request.preprocessing or [],
        "description": request.description or "",
        "created_at": templates.get(request.name, {}).get("created_at", now),
        "updated_at": now,
    }

    _save_user_templates(templates)

    return JSONResponse(
        status_code=200 if is_update else 201,
        content={
            "name": request.name,
            "created": not is_update,
            "updated": is_update,
        },
    )


@router.get("/user-templates/{name}")
def get_user_template(name: str) -> JSONResponse:
    """
    Get a specific user-defined template.
    """
    templates = _load_user_templates()
    if name not in templates:
        raise HTTPException(status_code=404, detail=f"User template '{name}' not found")

    data = templates[name]

    # Resolve dependencies for display
    try:
        config = PipelineConfig(
            steps=data.get("steps", []),
            preprocessing=data.get("preprocessing", []),
        )
        resolved = config.resolve_dependencies()
    except ValueError:
        resolved = data.get("steps", [])

    return JSONResponse(
        content={
            "name": name,
            "steps": data.get("steps", []),
            "preprocessing": data.get("preprocessing", []),
            "description": data.get("description", ""),
            "resolved_steps": resolved,
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
        }
    )


@router.delete("/user-templates/{name}")
def delete_user_template(name: str) -> JSONResponse:
    """
    Delete a user-defined template.
    """
    templates = _load_user_templates()
    if name not in templates:
        raise HTTPException(status_code=404, detail=f"User template '{name}' not found")

    del templates[name]
    _save_user_templates(templates)

    return JSONResponse(content={"deleted": name})
