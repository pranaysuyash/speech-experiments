"""
Pipeline Configuration - Dynamic step selection and pipeline templates.

Allows users to:
1. Define custom pipelines with any subset of steps
2. Configure preprocessing chains dynamically
3. Override model/step parameters per-run
4. Save and reuse pipeline templates

Example pipeline config:
    {
        "name": "fast_diarization",
        "description": "Quick speaker identification without full ASR",
        "steps": ["ingest", "diarization"],
        "preprocessing": ["trim_silence", "normalize_loudness"],
        "config": {
            "diarization": {"model_name": "pyannote_diarization"}
        }
    }
"""
from __future__ import annotations

import json
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


# All available steps in the system with their dependencies
STEP_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ingest": {
        "deps": [],
        "description": "Audio normalization and preprocessing",
        "required": True,  # Always runs first
        "produces": ["processed_audio"],
    },
    "asr": {
        "deps": ["ingest"],
        "description": "Speech-to-text transcription",
        "produces": ["transcript"],
        "config_schema": {
            "model_type": {"type": "string", "default": "faster_whisper"},
            "model_name": {"type": "string", "default": "large-v3"},
            "language": {"type": "string", "default": "en"},
        },
    },
    "diarization": {
        "deps": ["ingest"],
        "description": "Speaker identification and segmentation",
        "produces": ["diarization"],
        "config_schema": {
            "model_name": {"type": "string", "default": "heuristic_diarization"},
        },
    },
    "alignment": {
        "deps": ["asr", "diarization"],
        "description": "Merge transcription with speaker labels",
        "produces": ["alignment"],
    },
    "chapters": {
        "deps": ["alignment"],
        "description": "Topic segmentation using embeddings",
        "produces": ["chapters"],
    },
    "summarize_by_speaker": {
        "deps": ["alignment"],
        "description": "Per-speaker summary using LLM",
        "produces": ["summary"],
        "config_schema": {
            "model": {"type": "string", "default": "gpt-4o-mini"},
        },
    },
    "action_items_assignee": {
        "deps": ["alignment"],
        "description": "Extract action items with assignees",
        "produces": ["action_items"],
    },
    "bundle": {
        "deps": ["ingest"],  # Minimal dep; collects whatever is available
        "description": "Package outputs as Meeting Pack",
        "produces": ["bundle"],
    },
}

# Available preprocessing operators
PREPROCESSING_REGISTRY: Dict[str, Dict[str, Any]] = {
    "trim_silence": {
        "description": "Remove silence from start/end",
        "params": {
            "min_silence_ms": {"type": "int", "default": 500},
            "threshold_db": {"type": "float", "default": -40.0},
        },
    },
    "normalize_loudness": {
        "description": "LUFS loudness normalization",
        "params": {
            "target_lufs": {"type": "float", "default": -23.0},
        },
    },
    "normalize_volume": {
        "description": "Peak/RMS volume normalization",
        "params": {
            "target_db": {"type": "float", "default": -3.0},
            "method": {"type": "string", "default": "peak"},
        },
    },
    "resample": {
        "description": "Change sample rate",
        "params": {
            "target_sr": {"type": "int", "default": 16000},
        },
    },
    "extract_channel": {
        "description": "Extract mono channel from stereo",
        "params": {
            "channel": {"type": "int", "default": 0},
        },
    },
    "denoise": {
        "description": "Reduce background noise",
        "params": {
            "strength": {"type": "float", "default": 0.5},
        },
    },
    "speed": {
        "description": "Adjust playback speed",
        "params": {
            "factor": {"type": "float", "default": 1.0},
        },
    },
}


@dataclass
class PipelineConfig:
    """Configuration for a dynamic pipeline run."""
    
    name: str = "custom"
    description: str = ""
    
    # Steps to run (in dependency order)
    steps: List[str] = field(default_factory=lambda: ["ingest", "asr"])
    
    # Preprocessing operators to apply (in order)
    preprocessing: List[str] = field(default_factory=list)
    
    # Per-step configuration overrides
    config: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Device preference
    device_preference: List[str] = field(default_factory=lambda: ["mps", "cuda", "cpu"])
    
    def __post_init__(self):
        # Validate steps exist
        for step in self.steps:
            if step not in STEP_REGISTRY:
                raise ValueError(f"Unknown step: {step}. Available: {list(STEP_REGISTRY.keys())}")
        
        # Validate preprocessing ops exist
        for op in self.preprocessing:
            # Handle parameterized ops like "trim_silence(min_silence_ms=300)"
            op_name = op.split("(")[0]
            if op_name not in PREPROCESSING_REGISTRY:
                raise ValueError(f"Unknown preprocessing op: {op_name}. Available: {list(PREPROCESSING_REGISTRY.keys())}")
    
    def resolve_dependencies(self) -> List[str]:
        """
        Return steps in correct execution order, adding missing dependencies.
        
        If user requests [diarization], this returns [ingest, diarization].
        If user requests [alignment], this returns [ingest, asr, diarization, alignment].
        """
        resolved: List[str] = []
        seen: Set[str] = set()
        
        def add_with_deps(step: str):
            if step in seen:
                return
            
            # Add dependencies first
            step_info = STEP_REGISTRY.get(step, {})
            for dep in step_info.get("deps", []):
                add_with_deps(dep)
            
            seen.add(step)
            resolved.append(step)
        
        # Always start with ingest
        add_with_deps("ingest")
        
        # Add requested steps with their dependencies
        for step in self.steps:
            add_with_deps(step)
        
        return resolved
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        return cls(**data)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_json(cls, path: Path) -> "PipelineConfig":
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False)


# Built-in pipeline templates
PIPELINE_TEMPLATES: Dict[str, PipelineConfig] = {
    "ingest_only": PipelineConfig(
        name="ingest_only",
        description="Audio preprocessing only - no transcription",
        steps=["ingest"],
        preprocessing=["trim_silence", "normalize_loudness"],
    ),
    "fast_asr": PipelineConfig(
        name="fast_asr",
        description="Quick transcription without diarization",
        steps=["ingest", "asr"],
        preprocessing=["normalize_loudness"],
    ),
    "asr_diarization": PipelineConfig(
        name="asr_diarization",
        description="Transcription with speaker identification",
        steps=["ingest", "asr", "diarization", "alignment"],
        preprocessing=["trim_silence", "normalize_loudness"],
    ),
    "diarization_only": PipelineConfig(
        name="diarization_only",
        description="Speaker segmentation without transcription",
        steps=["ingest", "diarization"],
        preprocessing=["normalize_loudness"],
    ),
    "full_meeting": PipelineConfig(
        name="full_meeting",
        description="Complete meeting analysis pipeline",
        steps=["ingest", "asr", "diarization", "alignment", "chapters", "summarize_by_speaker", "action_items_assignee", "bundle"],
        preprocessing=["trim_silence", "normalize_loudness"],
    ),
    "quick_summary": PipelineConfig(
        name="quick_summary",
        description="Fast transcription and summary (no diarization)",
        steps=["ingest", "asr", "summarize_by_speaker"],
        preprocessing=["normalize_loudness"],
        config={
            "asr": {"model_type": "faster_whisper", "model_name": "base"},
        },
    ),
}


def get_pipeline_template(name: str) -> Optional[PipelineConfig]:
    """Get a built-in pipeline template by name."""
    return PIPELINE_TEMPLATES.get(name)


def list_pipeline_templates() -> List[Dict[str, Any]]:
    """List all available pipeline templates."""
    return [
        {
            "name": name,
            "description": cfg.description,
            "steps": cfg.steps,
            "preprocessing": cfg.preprocessing,
        }
        for name, cfg in PIPELINE_TEMPLATES.items()
    ]


def list_available_steps() -> List[Dict[str, Any]]:
    """List all available steps with their metadata."""
    return [
        {
            "name": name,
            "deps": info.get("deps", []),
            "description": info.get("description", ""),
            "produces": info.get("produces", []),
            "config_schema": info.get("config_schema"),
        }
        for name, info in STEP_REGISTRY.items()
    ]


def list_preprocessing_ops() -> List[Dict[str, Any]]:
    """List all available preprocessing operators."""
    return [
        {
            "name": name,
            "description": info.get("description", ""),
            "params": info.get("params", {}),
        }
        for name, info in PREPROCESSING_REGISTRY.items()
    ]


def validate_pipeline_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate a pipeline configuration and return list of errors.
    
    Returns empty list if valid.
    """
    errors = []
    
    # Check steps
    steps = config.get("steps", [])
    if not steps:
        errors.append("Pipeline must have at least one step")
    
    for step in steps:
        if step not in STEP_REGISTRY:
            errors.append(f"Unknown step: {step}")
    
    # Check preprocessing
    for op in config.get("preprocessing", []):
        op_name = op.split("(")[0]
        if op_name not in PREPROCESSING_REGISTRY:
            errors.append(f"Unknown preprocessing operator: {op_name}")
    
    # Validate step config keys
    step_config = config.get("config", {})
    for step_name, step_cfg in step_config.items():
        if step_name not in STEP_REGISTRY:
            errors.append(f"Config for unknown step: {step_name}")
    
    return errors
