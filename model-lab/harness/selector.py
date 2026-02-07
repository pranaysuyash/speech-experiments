"""
Model selector utilities (LCS-Z).

Filter models by device, runtime, surface, CI compatibility.
"""

from __future__ import annotations

from typing import Any, Literal


try:
    from harness.registry import ModelRegistry
    REGISTRY_AVAILABLE = True
except ImportError:
    ModelRegistry = None
    REGISTRY_AVAILABLE = False


def list_models_by_filter(
    *,
    device: str | None = None,
    runtime: str | None = None,
    surface: str | None = None,
    ci: bool | None = None,
) -> list[dict[str, Any]]:
    """
    Filter models by various criteria.
    
    Args:
        device: Filter by hardware support (cpu, cuda, mps)
        runtime: Filter by runtime (pytorch, nemo, onnx, etc)
        surface: Filter by capability surface (asr, asr_stream, tts, etc)
        ci: Filter by CI compatibility (True = ci safe, False = not ci safe)
    
    Returns:
        List of matching model metadata dicts with model_id added.
    
    Example:
        >>> list_models_by_filter(device="mps", surface="asr_stream")
        [{"model_id": "kyutai_streaming", "capabilities": ["asr_stream"], ...}]
    """
    if not REGISTRY_AVAILABLE:
        return []
    
    models = ModelRegistry.list_models()
    results = []
    
    for model_id in models:
        try:
            meta = ModelRegistry.get_model_metadata(model_id)
        except Exception:
            continue
        
        # Add model_id to metadata
        meta["model_id"] = model_id
        
        # Filter by device
        if device is not None:
            hardware = meta.get("hardware", [])
            if device not in hardware:
                continue
        
        # Filter by surface (capability)
        if surface is not None:
            capabilities = meta.get("capabilities", [])
            if surface not in capabilities:
                continue
        
        # Filter by CI compatibility requires claims file
        if ci is not None:
            # Default to True if not specified
            model_ci = _get_model_ci_flag(model_id)
            if ci and not model_ci:
                continue
            if not ci and model_ci:
                continue
        
        # Filter by runtime requires claims file
        if runtime is not None:
            model_runtime = _get_model_runtime(model_id)
            if model_runtime != runtime:
                continue
        
        results.append(meta)
    
    return results


def _get_model_ci_flag(model_id: str) -> bool:
    """Get CI flag from model claims. Returns True if ci: true or not specified."""
    import yaml
    from pathlib import Path
    
    claims_path = Path(__file__).parent.parent / "models" / model_id / "claims.yaml"
    if not claims_path.exists():
        return True  # Assume CI-safe if no claims
    
    try:
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        return claims.get("ci", True)
    except Exception:
        return True


def _get_model_runtime(model_id: str) -> str | None:
    """Get runtime from model claims."""
    import yaml
    from pathlib import Path
    
    claims_path = Path(__file__).parent.parent / "models" / model_id / "claims.yaml"
    if not claims_path.exists():
        return None
    
    try:
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        return claims.get("runtime")
    except Exception:
        return None


def format_model_table(models: list[dict[str, Any]]) -> str:
    """Format models as a simple table."""
    if not models:
        return "No models match the filter."
    
    lines = [
        "| Model ID | Capabilities | Hardware |",
        "|----------|--------------|----------|",
    ]
    
    for m in models:
        model_id = m.get("model_id", "?")
        caps = ", ".join(m.get("capabilities", []))
        hardware = ", ".join(m.get("hardware", []))
        lines.append(f"| {model_id} | {caps} | {hardware} |")
    
    return "\n".join(lines)


def get_streaming_models(device: str | None = None) -> list[str]:
    """Get all streaming ASR models, optionally filtered by device."""
    models = list_models_by_filter(surface="asr_stream", device=device)
    return [m["model_id"] for m in models]


def get_ci_safe_models(surface: str | None = None) -> list[str]:
    """Get all CI-safe models, optionally filtered by surface."""
    models = list_models_by_filter(ci=True, surface=surface)
    return [m["model_id"] for m in models]


def get_models_by_runtime(runtime: str, device: str | None = None) -> list[str]:
    """Get models by runtime (pytorch, nemo, onnx, etc)."""
    models = list_models_by_filter(runtime=runtime, device=device)
    return [m["model_id"] for m in models]
