"""
Diarization Runner - Core Logic.

Exposes `run_diarization` for use by SessionRunner and scripts.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

from harness.registry import ModelRegistry
from harness.run_provenance import create_provenance
from harness.runner_schema import (
    RunnerArtifact, InputsSchema, QualityMetrics, create_run_context
)
from harness.media_ingest import ingest_media
from harness.preprocess_ops import run_preprocessing_chain

logger = logging.getLogger("diarization")

def run_diarization(
    input_path: Path,
    model_name: str,
    output_dir: Path,
    device: str = "cpu",
    pre_ops: Optional[str] = None,
    dataset_def: Optional[Dict[str, Any]] = None,
    force: bool = False
) -> Path:
    """
    Run Diarization on input media.
    
    Args:
        input_path: Path to media file.
        model_name: Name of model to use.
        output_dir: Directory to save artifact.
        device: 'cpu' or 'cuda'/'mps'.
        pre_ops: Preprocessing operations string.
        dataset_def: Optional golden dataset definition.
        force: Overwrite existing artifact.
        
    Returns:
        Path to the generated Diarization artifact JSON.
    """
    input_path = input_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load already-processed audio (input_path is from the ingest step)
    import soundfile as sf
    audio, sr = sf.read(str(input_path))
    
    # Compute hash for provenance
    import hashlib
    audio_hash = hashlib.sha256(audio.tobytes()).hexdigest()[:16]
    
    # Preprocessing
    audio_for_model = audio
    sr_for_model = sr
    
    if pre_ops:
         results = run_preprocessing_chain(audio_for_model, sr_for_model, pre_ops.split(','))
         if results:
             audio_for_model = results[-1].audio
             sr_for_model = results[-1].sample_rate

    # 2. Load Model via ModelRegistry
    try:
        bundle = ModelRegistry.load_model(model_name, {}, device)
    except Exception as e:
        raise ValueError(f"Failed to load diarization model {model_name}: {e}")
        
    # 3. Run Diarization via bundle's diarization capability
    t0 = time.time()
    diarize_func = bundle.get("diarization", {}).get("diarize")
    if not diarize_func:
        # Fallback: return empty result if model doesn't support diarization
        logger.warning(f"Model {model_name} doesn't have diarize capability")
        segments = []
    else:
        result = diarize_func(audio_for_model, sr_for_model)
        segments = result.get("segments", [])
        
    duration_ms = int((time.time() - t0) * 1000)
    
    # 4. Segments already extracted from result, proceed to artifact creation

    # 5. Create Artifact
    artifact_name = f"diarization_{model_name}.json"
    artifact_path = output_dir / artifact_name
    
    run_ctx = create_run_context(
        task="diarization",
        model_id=model_name,
        grade="adhoc",
        device=device,
        model_version=model_name,
    )
    
    prov = create_provenance(dataset_id="session")
    
    inputs = InputsSchema(
        audio_path=str(input_path),
        audio_hash=audio_hash,
    )
    
    output_data = {
        "segments": segments,
        "num_speakers": len(set(s['speaker'] for s in segments))
    }
    
    artifact = RunnerArtifact(
        run_context=run_ctx,
        inputs=inputs,
        metrics_quality=QualityMetrics(),
        metrics_structural={"duration_ms": duration_ms},
        provenance=prov if isinstance(prov, dict) else {}, # legacy compat
        output=output_data,
    )
    
    with open(artifact_path, 'w') as f:
        json.dump(artifact.to_dict(), f, indent=2)
        
    return artifact_path
