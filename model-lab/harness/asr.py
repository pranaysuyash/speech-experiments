"""
ASR Runner - Core Logic.

Exposes `run_asr` for use by SessionRunner and scripts.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union

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

def run_asr(
    input_path: Path,
    output_dir: Path,
    config: Optional[Dict[str, Any]] = None
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
    config = config or {}
    model_type = config.get("model_type", "faster_whisper")
    # model_name might be used by the loader for weights/size
    model_name = config.get("model_name", "default")
    device = config.get("device", "cpu")
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
    
    try:
        model_bundle = ModelRegistry.load_model(model_type, config, device=device)
    except Exception as e:
        raise ValueError(f"Failed to load model {model_type}: {e}")

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
    try:
         result = transcribe_func(audio_data, sr=sr)
    except Exception as e:
         raise RuntimeError(f"Transcription failed: {e}")

    duration_ms = int((time.time() - t0) * 1000)
    
    # 5. Metrics (if golden) - Skipped for now
    metrics = None
    diagnostics = []
        
    # 6. Create Artifact
    artifact_name = f"asr_{model_type}_{model_name}.json"
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
        
    return {
        "artifacts": [
             {"type": "asr_json", "path": str(artifact_path), "hash": sha256_file(artifact_path)}
        ],
        "result": output_data # Store minimal result if needed
    }

import time
