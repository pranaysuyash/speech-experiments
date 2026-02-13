"""
ASR Runner - Core Logic.

Exposes `run_asr` for use by SessionRunner and scripts.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Union, Callable

from harness.audio_io import AudioLoader
from harness.registry import ModelRegistry
from harness.metrics_asr import ASRMetrics, diagnose_output_quality
from harness.run_provenance import create_provenance
from harness.runner_schema import (
    RunnerArtifact, RunContext, InputsSchema, QualityMetrics,
    compute_pcm_hash, create_run_context
)
from harness.media_ingest import ingest_media, IngestResult, IngestConfig, sha256_file
from harness.preprocess_ops import run_preprocessing_chain

logger = logging.getLogger("asr")


@dataclass
class ResolvedASRConfig:
    """
    Ground truth of what WILL execute.
    Resolved BEFORE model loading, persisted BEFORE execution.
    """
    model_id: str      # Exact model name (not alias like "default")
    source: str        # "local" | "hf" | "api"
    device: str        # Actual execution device: "cpu" | "cuda" | "mps"
    device: str        # Actual execution device: "cpu" | "cuda" | "mps"
    reason: str        # Reason for device/model choice
    language: str      # "auto" | specific language code

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def resolve_asr_config(user_config: Optional[Dict[str, Any]] = None) -> ResolvedASRConfig:
    """
    Pure resolution function - determines what WILL execute.
    
    This function MUST be called BEFORE model loading.
    The result MUST be persisted BEFORE execution begins.
    
    Resolution rules:
    - model_id: "default" -> actual model name from registry
    - source: Determined by model type and availability
    - device: Resolved based on availability (mps/cuda/cpu)
    - language: "auto" unless forced in config
    """
    user_config = user_config or {}

    # 1. Resolve model_id
    model_type = user_config.get("model_type", "faster_whisper")
    # Accept both model_name and model_size for flexibility (UI uses model_size)
    model_name = user_config.get("model_name") or user_config.get("model_size", "default")

    # Map "default" to actual model name per model type
    if model_name == "default":
        default_models = {
            "faster_whisper": "large-v3",
            "whisper": "large-v3",
            "distil_whisper": "distil-whisper/distil-large-v3",
            "lfm2_5_audio": "LiquidAI/LFM2.5-Audio-1.5B",
            "seamlessm4t": "facebook/seamless-m4t-v2-large",
        }
        model_name = default_models.get(model_type, "large-v3")
    
    model_id = f"{model_type}:{model_name}"
    
    # 2. Resolve source
    # For now, simple heuristic based on model type
    # Could be enhanced to check local cache vs HF download
    if model_type in ("faster_whisper", "whisper"):
        source = "hf"  # Downloads from HuggingFace
    elif model_type == "whisper_cpp":
        source = "local"  # Uses local ggml file
    elif model_type in ("lfm2_5_audio", "distil_whisper", "seamlessm4t"):
        source = "hf"
    else:
        source = "unknown"
    
    # 3. Resolve device
    # P4: Device Selection Contract
    # Priority: device_preference list > device string > default ["cpu"]
    preference = user_config.get("device_preference")
    if not preference:
        preference = [user_config.get("device", "cpu")]
    
    actual_device = None
    reason = None
    
    for cand in preference:
        # Check Model-Specific Constraints
        if model_type == "faster_whisper" and cand == "mps":
            # Constraint: faster_whisper doesn't support MPS yet
            continue
            
        if model_type == "lfm2_5_audio" and cand == "cuda":
            # Constraint: LFM has known CUDA issues
            continue
            
        # If we reached here, the device is valid for the model configuration
        # (We assume system availability is handled by the worker runtime or assumed available if requested)
        actual_device = cand
        reason = f"preference_{cand}"
        break
    
    # Final cleanup if loop exhausted without selection (unlikely if 'cpu' in list, but possible)
    if not actual_device:
        actual_device = "cpu"
        reason = "exhausted_preference"
    
    # 4. Resolve language
    language = user_config.get("language", "auto")
    
    return ResolvedASRConfig(
        model_id=model_id,
        source=source,
        device=actual_device,
        reason=reason,
        language=language
    )


def run_asr(
    input_path: Path,
    output_dir: Path,
    config: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[], None]] = None,
    update_progress: Optional[Callable[[int, Optional[str], Optional[int]], None]] = None
) -> Dict[str, Any]:
    """
    Run ASR on input media.
    
    Args:
        input_path: Path to media file (or processed audio).
        output_dir: Directory to save artifact.
        config: Configuration dictionary (model_name, device, etc).
        
    Returns:
        Dictionary containing input/output/artifacts info.
    """
    input_path = input_path.resolve()
    
    # 0. Resolve Configuration (Crucial for Device/Model Selection)
    resolved = resolve_asr_config(config)
    model_type = config.get("model_type", "faster_whisper") # Still need type for loader? resolve_asr_config output has model_id "type:name"
    # Actually ModelRegistry takes (model_type, config, device).
    # We should respect resolved values.
    # User config might have "model_name": "default" -> resolved "large-v3"
    # User config "device": "mps" -> resolved "cpu" (if fw)
    
    device = resolved.device
    model_id = resolved.model_id # type:name
    if ":" in model_id:
        _, model_name = model_id.split(":", 1)
    else:
        model_name = "unknown"
    
    # Parse back? Or just rely on config?
    # ModelRegistry needs type/config. 
    # Let's pass 'device' explicitly.

    pre_ops = config.get("pre_ops")
    dataset_def = config.get("dataset_def")
    
    input_path = input_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Ingest (or use direct if already wav/pcm?)
    # For robust handling, we re-ingest (cheap if hash matches?)
    # But if input is from session ingest, it's already a WAV.
    # ingest_media re-verifies.
    # We'll use ingest_media for consistency of hashing, but maybe skip filters if trusted?
    # SessionRunner passes 'ingest' artifacts usually.
    ingest = ingest_media(input_path, output_dir, IngestConfig()) # Use default config for re-ingest? 
    # NOTE: If we use default config here, we might double-process if session used specific config.
    # But since input_path IS the processed audio from session ingest (usually),
    # doing default ingest on it (no norm, no trim) is basically a pass-through identity check + hash.
    # That is acceptable for now.
    
    # 2. Preprocessing (Additional)
    audio_for_model = ingest["processed_audio_path"] # It's a string path
    ingest_audio_path = Path(audio_for_model)
    audio_hash = ingest["audio_content_hash"]
    
    # 3. Load Model
    # ModelRegistry.load_model returns a Bundle dict
    # We don't use context manager pattern here anymore for the bundle itself.
    
    # 3. Load Model
    # ModelRegistry.load_model returns a Bundle dict
    # We don't use context manager pattern here anymore for the bundle itself.
    
    t0 = time.time()
    logger.info(f"Loading {model_type} on {device}")
    
    if update_progress:
        update_progress(5, "Loading model...")
    
    try:
        model_bundle = ModelRegistry.load_model(model_type, config, device=device)
    except Exception as e:
        raise ValueError(f"Failed to load model {model_type}: {e}")

    if update_progress:
        update_progress(20, "Model loaded, preparing audio...")

    # Check capabilities
    caps = model_bundle.get("capabilities", [])
    if "asr" not in caps:
         raise RuntimeError(f"Model {model_type} does not support ASR")
         
    transcribe_func = model_bundle["asr"]["transcribe"]
    
    # Load audio using soundfile
    # We rely on soundfile to handle various formats if ingest path is passed
    import soundfile as sf
    try:
        audio_data, sr = sf.read(ingest_audio_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read audio file {ingest_audio_path}: {e}")
    
    logger.info(f"Transcribing {ingest_audio_path.name}...")
    if update_progress:
        update_progress(30, "Transcribing audio...")
    try:
         # Pass progress_callback if supported? 
         # Bundle Contract v1 doesn't strictly mandate progress_callback in signature, 
         # but we can pass it in kwargs.
         result = transcribe_func(audio_data, sr=sr, progress_callback=progress_callback)
    except TypeError:
         # Fallback for models not supporting progress_callback
         logger.warning(f"Model {model_type} does not support progress tracking")
         result = transcribe_func(audio_data, sr=sr)
    except Exception as e:
         raise RuntimeError(f"Transcription failed: {e}")

    if update_progress:
        update_progress(90, "Processing results...")

    duration_ms = int((time.time() - t0) * 1000)
    
    # 5. Metrics (if golden) - Skipped for now
    metrics = None
    diagnostics = []
        
    # Sanitize model name for filename
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    artifact_name = f"asr_{model_type}_{safe_model_name}.json"
    artifact_path = output_dir / artifact_name
    
    # RunContext
    run_ctx = create_run_context(
        task="asr",
        model_id=f"{model_type}:{model_name}",
        grade="adhoc",
        device=device
    )
    
    # Provenance
    prov = create_provenance(
         dataset_id="session"
    )
    
    # Inputs
    inputs = InputsSchema(
        audio_hash=audio_hash,
        audio_path=str(ingest_audio_path), # Used to be media_path
        # Runner schema validation requires audio_hash and audio_path.
    )
    
    # Output schema:
    # We want standard ASR output.
    output_data = {
        "text": result.get("text", ""),
        "segments": result.get("segments", []),
        "language": result.get("language", "en"),
        "meta": result.get("meta", {})
    }
    
    # RTF Calculation
    audio_duration = ingest.get("duration", 0) # IngestResult uses 'duration' or 'duration_s'? Check IngestResult.
    # IngestResult (Step 1937) has 'duration_s'.
    # ingest() returns dict. Let's assume 'duration_s' exists.
    audio_duration_s = ingest.get("duration_s", 0.1)
    rtf = (duration_ms / 1000.0) / audio_duration_s if audio_duration_s > 0 else 0.0

    artifact = RunnerArtifact(
        run_context=run_ctx,
        inputs=inputs,
        provenance=prov,
        output=output_data,
        metrics_quality=QualityMetrics(), # Default empty for adhoc
        metrics_structural={
            "duration_ms": duration_ms,
            "rtf": rtf,
            "latency_ms": duration_ms
        },
        gates={}
    )
    
    with open(artifact_path, 'w') as f:
        json.dump(artifact.to_dict(), f, indent=2)
    
    if update_progress:
        update_progress(100, "ASR completed")
        
    return {
        "artifacts": [
             {"type": "asr_json", "path": str(artifact_path), "hash": sha256_file(artifact_path)}
        ],
        "result": output_data,
        "resolved_config": resolved.to_dict(),
        "requested_config": config
    }

import time
