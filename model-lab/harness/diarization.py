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
from harness.run_provenance import create_provenance, create_run_context
from harness.runner_schema import (
    RunnerArtifact, InputsSchema
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
    
    # 1. Ingest
    ingest = ingest_media(input_path)
    
    # Preprocessing
    audio_for_model = ingest.audio
    sr_for_model = ingest.sample_rate
    
    if pre_ops:
         results = run_preprocessing_chain(audio_for_model, sr_for_model, pre_ops.split(','))
         if results:
             audio_for_model = results[-1].audio
             sr_for_model = results[-1].sample_rate

    # 2. Load Model
    loader = ModelRegistry.get_loader(model_name)
    if not loader:
        raise ValueError(f"Model {model_name} not found in registry")
        
    # 3. Run Diarization
    t0 = time.time()
    with loader.load(device=device) as model:
        # Assuming model.diarize returns a simplified result object or Pyannote Annotation
        # In existing scripts, we convert result to list of segments.
        result = model.diarize(audio_for_model, sr_for_model)
        
    duration_ms = int((time.time() - t0) * 1000)
    
    # 4. Process Output
    # Convert result to standard list of segments: {start, end, speaker}
    # If result has .iter_tracks() (pyannote):
    segments = []
    if hasattr(result, "itertracks"):
        for turn, _, speaker in result.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
    elif isinstance(result, list): # Heuristic might return list
        segments = result
    else:
        # Fallback or error?
        logger.warning(f"Unknown diarization result type: {type(result)}")
        segments = []

    # 5. Create Artifact
    artifact_name = f"diarization_{model_name}.json"
    artifact_path = output_dir / artifact_name
    
    run_ctx = create_run_context(
        task="diarization",
        model_id=model_name,
        dataset_id="session",
        compute_device=device
    )
    
    prov = create_provenance(dataset_id="session")
    
    inputs = InputsSchema(
        audio_hash=ingest.audio_hash,
        media_path=str(input_path)
    )
    
    output_data = {
        "segments": segments,
        "num_speakers": len(set(s['speaker'] for s in segments))
    }
    
    artifact = RunnerArtifact(
        run_context=run_ctx,
        inputs=inputs,
        provenance=prov,
        output=output_data,
        metrics=None
    )
    
    with open(artifact_path, 'w') as f:
        json.dump(artifact.to_dict(), f, indent=2)
        
    return artifact_path
