"""
Pipelines API - Dynamic step selection and pipeline management.

Provides endpoints for:
- Listing available steps and preprocessing operators
- Listing and retrieving pipeline templates
- Creating custom pipeline configurations
- Running ad-hoc pipelines with user-selected steps
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from harness.pipeline_config import (
    PipelineConfig,
    STEP_REGISTRY,
    PREPROCESSING_REGISTRY,
    PIPELINE_TEMPLATES,
    list_available_steps,
    list_preprocessing_ops,
    list_pipeline_templates,
    validate_pipeline_config,
    get_pipeline_template,
)

router = APIRouter(prefix="/api/pipelines", tags=["pipelines"])


# ============================================================================
# Request/Response Models
# ============================================================================

class PipelineRunRequest(BaseModel):
    """Request body for creating a dynamic pipeline run."""
    
    # Either use a template name OR provide custom steps
    template: Optional[str] = None
    
    # Custom step selection (if not using template)
    steps: Optional[List[str]] = None
    
    # Preprocessing operators to apply
    preprocessing: Optional[List[str]] = None
    
    # Per-step configuration overrides
    config: Optional[Dict[str, Dict[str, Any]]] = None
    
    # Device preference
    device_preference: Optional[List[str]] = None
    
    # Run metadata
    name: Optional[str] = None
    use_case_id: Optional[str] = None


class PipelineValidateRequest(BaseModel):
    """Request body for validating a pipeline configuration."""
    steps: List[str]
    preprocessing: Optional[List[str]] = None
    config: Optional[Dict[str, Dict[str, Any]]] = None


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


@router.get("/templates/{name}")
def get_pipeline_template_detail(name: str) -> JSONResponse:
    """
    Get details of a specific pipeline template.
    """
    template = get_pipeline_template(name)
    if not template:
        raise HTTPException(
            status_code=404,
            detail=f"Template '{name}' not found. Available: {list(PIPELINE_TEMPLATES.keys())}"
        )
    
    return JSONResponse(content={
        **template.to_dict(),
        "resolved_steps": template.resolve_dependencies(),
    })


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
        return JSONResponse(content={
            "valid": False,
            "errors": errors,
            "resolved_steps": [],
        })
    
    try:
        config = PipelineConfig(
            steps=request.steps,
            preprocessing=request.preprocessing or [],
            config=request.config or {},
        )
        resolved = config.resolve_dependencies()
        
        return JSONResponse(content={
            "valid": True,
            "errors": [],
            "resolved_steps": resolved,
        })
    except ValueError as e:
        return JSONResponse(content={
            "valid": False,
            "errors": [str(e)],
            "resolved_steps": [],
        })


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
        
        return JSONResponse(content={
            "requested_steps": request.steps,
            "resolved_steps": resolved,
            "added_dependencies": [s for s in resolved if s not in request.steps],
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/step/{name}")
def get_step_info(name: str) -> JSONResponse:
    """
    Get detailed information about a specific step.
    """
    if name not in STEP_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Step '{name}' not found. Available: {list(STEP_REGISTRY.keys())}"
        )
    
    info = STEP_REGISTRY[name]
    
    # Find which steps depend on this one
    dependents = [
        s for s, i in STEP_REGISTRY.items()
        if name in i.get("deps", [])
    ]
    
    return JSONResponse(content={
        "name": name,
        "deps": info.get("deps", []),
        "dependents": dependents,
        "description": info.get("description", ""),
        "produces": info.get("produces", []),
        "config_schema": info.get("config_schema"),
        "required": info.get("required", False),
    })


@router.get("/preprocessing/{name}")
def get_preprocessing_info(name: str) -> JSONResponse:
    """
    Get detailed information about a preprocessing operator.
    """
    if name not in PREPROCESSING_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Operator '{name}' not found. Available: {list(PREPROCESSING_REGISTRY.keys())}"
        )
    
    info = PREPROCESSING_REGISTRY[name]
    
    return JSONResponse(content={
        "name": name,
        "description": info.get("description", ""),
        "params": info.get("params", {}),
    })


# ============================================================================
# Helper for creating pipeline runs (used by workbench endpoint)
# ============================================================================

def build_pipeline_config(
    template: Optional[str] = None,
    steps: Optional[List[str]] = None,
    preprocessing: Optional[List[str]] = None,
    config: Optional[Dict[str, Dict[str, Any]]] = None,
    device_preference: Optional[List[str]] = None,
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
