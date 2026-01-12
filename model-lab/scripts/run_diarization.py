#!/usr/bin/env python3
"""
Run Diarization evaluation for a specific model and dataset.

Usage:
    uv run scripts/run_diarization.py --model pyannote_diarization --dataset diar_smoke_v1
    uv run scripts/run_diarization.py --model pyannote_diarization --audio file.wav  # Adhoc
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

import yaml
import soundfile as sf
import numpy as np
import torch

# Add harness to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.registry import ModelRegistry
from harness.metrics_diarization import DiarizationMetrics
from harness.taxonomy import TaskType, EvidenceGrade
from harness.run_provenance import create_provenance, create_run_context, RUN_SCHEMA_VERSION
from harness.adhoc_dataset import create_adhoc_dataset
from harness.runner_schema import (
    RunnerArtifact, RunContext, InputsSchema, QualityMetrics,
    validate_artifact, compute_pcm_hash,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_diarization")


def load_dataset(dataset_id: str) -> Dict[str, Any]:
    """Load dataset definition."""
    # Try golden first
    path = Path(f"data/golden/{dataset_id}.yaml")
    if not path.exists():
        # Try ad-hoc location if needed (omitted for now)
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    with open(path) as f:
        return yaml.safe_load(f)


def run_diarization_adhoc(model_id: str, audio_path: str, device: str = "cpu"):
    """
    Run diarization on a single audio file (adhoc mode).
    
    Produces artifact with grade=adhoc and structural metrics only.
    No DER (requires reference RTTM).
    """
    audio_path = Path(audio_path).resolve()
    logger.info(f"=== Diarization Adhoc: {model_id} on {audio_path.name} ===")
    
    # Create adhoc dataset metadata
    adhoc = create_adhoc_dataset(audio_path, task="diarization")
    logger.info(f"Dataset ID: {adhoc.dataset_id}")
    
    # Load model
    config_path = Path(f"models/{model_id}/config.yaml")
    if not config_path.exists():
        raise ValueError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    bundle = ModelRegistry.load_model(config['model_type'], config, device=device)
    
    if "diarization" not in bundle:
        raise ValueError(f"Model {model_id} does not support 'diarization'")
    
    diarize_fn = bundle["diarization"]["diarize"]
    logger.info(f"✓ Model loaded")
    
    # Load audio
    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    
    duration_s = len(audio) / sr
    
    # Run diarization
    t0 = time.time()
    output = diarize_fn(audio, sr=sr)
    latency = time.time() - t0
    
    rtf = latency / duration_s if duration_s > 0 else 0
    segments = output.get("segments", [])
    
    # Count speakers
    speakers = set()
    for seg in segments:
        if isinstance(seg, dict) and "speaker" in seg:
            speakers.add(seg["speaker"])
    num_speakers = len(speakers)
    
    # Create schema-validated artifact
    schema_run_context = RunContext(
        task="diarization",
        model_id=model_id,
        grade="adhoc",
        timestamp=datetime.now().isoformat(),
        git_hash=None,
        command=sys.argv,
        device=device,
    )
    
    schema_inputs = InputsSchema(
        audio_path=str(audio_path),
        audio_hash=compute_pcm_hash(audio),
        dataset_id=adhoc.dataset_id,
        audio_duration_s=duration_s,
        sample_rate=sr,
    )
    
    structural_metrics = {
        'rtf': rtf,
        'num_speakers_pred': num_speakers,
        'segment_count': len(segments),
        'latency_s': latency,
        'duration_s': duration_s,
    }
    
    schema_artifact = RunnerArtifact(
        run_context=schema_run_context,
        inputs=schema_inputs,
        metrics_quality=QualityMetrics(),  # All None - no reference RTTM
        metrics_structural=structural_metrics,
        output={
            'num_speakers': num_speakers,
            'segments': segments[:30],  # Limit stored segments
        },
        provenance={'has_ground_truth': False, 'metrics_valid': True},
        gates={'has_failure': False},
    )
    
    # Validate before writing
    validate_artifact(schema_artifact)
    
    # Save artifact
    runs_dir = Path(f'runs/{model_id}/diarization')
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    run_file = runs_dir / f'adhoc_{timestamp}.json'
    
    with open(run_file, 'w') as f:
        json.dump(schema_artifact.to_dict(), f, indent=2, default=str)
    
    logger.info(f"✓ Adhoc run completed: {num_speakers} speakers detected")
    print(f"ARTIFACT_PATH:{run_file}")
    return schema_artifact.to_dict(), str(run_file)


def save_run_artifact(model_id: str,
                      dataset_id: str,
                      dataset_path: Path,
                      results: List[Dict[str, Any]],
                      summary: Dict[str, Any],
                      output_dir: Path,
                      has_ground_truth: bool = False):
    """Save execution artifacts with provenance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    run_id = f"{dataset_id}_{int(time.time())}"
    
    # Create provenance EARLY
    provenance = create_provenance(
        dataset_id=dataset_id,
        dataset_path=dataset_path,
        audio_path=None,  # Multiple audio files
        ground_truth_path=None if not has_ground_truth else dataset_path,
        metrics_valid=True,
    )
    
    # Create run_context for interpretable metrics
    run_context = create_run_context(
        device=summary.get("device", "unknown"),
        audio_duration_s=summary.get("total_duration_s"),
    )
    
    artifact = {
        "meta": {
            "run_id": run_id,
            "timestamp": timestamp,
            "model_id": model_id,
            "task": "diarization"
        },
        "capability": "diarization",  # Required for task detection
        "input": {
            "dataset_id": dataset_id,
            "count": len(results)
        },
        "system": {
            "device": summary.get("device", "unknown")
        },
        "metrics": summary.get("metrics", {}),
        "gates": summary.get("gates", {}),
        "provenance": provenance,  # REQUIRED for all runs
        "run_context": run_context,  # REQUIRED for interpretable metrics
        "evidence": {
            "grade": "smoke" if "smoke" in dataset_id else "golden_batch",
            "dataset_id": dataset_id,
            "sanity_gates": summary.get("sanity_gates", {}),
            "wer_valid": True
        },
        "results": results
    }
    
    outfile = output_dir / f"{run_id}.json"
    with open(outfile, "w") as f:
        json.dump(artifact, f, indent=2)
    
    logger.info(f"Saved run artifact: {outfile}")


def main():
    parser = argparse.ArgumentParser(description="Run Diarization evaluation")
    parser.add_argument("--model", required=True, help="Model ID (e.g., pyannote_diarization)")
    
    # Mutually exclusive: --dataset or --audio
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--dataset", help="Dataset ID (e.g., diar_smoke_v1)")
    input_group.add_argument("--audio", type=Path, help="Single audio file (adhoc mode)")
    
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda, mps)")
    
    args = parser.parse_args()
    
    # Adhoc mode
    if args.audio:
        try:
            result, artifact_path = run_diarization_adhoc(args.model, str(args.audio), args.device)
            logger.info(f"🎉 Adhoc run completed! Artifact: {artifact_path}")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Adhoc run failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Dataset mode (original code path)
    
    # 1. Load Model
    try:
        config_path = Path(f"models/{args.model}/config.yaml")
        if not config_path.exists():
            raise ValueError(f"Config not found for model: {args.model}")
            
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        bundle = ModelRegistry.load_model(config['model_type'], config, device=args.device)
        
        if "diarization" not in bundle:
            raise ValueError(f"Model {args.model} does not support 'diarization' capability")
            
        diarize_fn = bundle["diarization"]["diarize"]
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
        
    # 2. Load Dataset
    try:
        dataset = load_dataset(args.dataset)
        cases = dataset.get("cases", [])
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)
        
    logger.info(f"Starting Diarization run for {args.model} on {args.dataset} ({len(cases)} cases)")
    
    results = []
    
    # 3. Process Cases
    for case in cases:
        case_id = case["id"]
        audio_file = Path(case["audio_file"])
        expected = case["expected"]
        
        if not audio_file.exists():
            logger.warning(f"Audio file missing: {audio_file}, skipping")
            continue
            
        try:
            # Load audio
            audio, sr = sf.read(str(audio_file))
            if audio.ndim > 1:
                audio = audio.mean(axis=1) # Mono mixdown
            audio = audio.astype(np.float32)
            
            logger.info(f"Processing {case_id} ({len(audio)/sr:.1f}s)...")
            
            # Run Inference
            t0 = time.time()
            output = diarize_fn(audio, sr=sr)
            latency = time.time() - t0
            
            # Compute Metrics
            eval_result = DiarizationMetrics.evaluate(
                pred_segments=output["segments"],
                expected=expected,
                latency_s=latency
            )
            
            # Store Result
            results.append({
                "id": case_id,
                "audio_file": str(audio_file),
                "duration_s": len(audio)/sr,
                "output": output,
                "metrics": {
                    "num_speakers_pred": eval_result.num_speakers_pred,
                    "num_speakers_auth": eval_result.num_speakers_auth,
                    "speaker_count_error": eval_result.speaker_count_error,
                    "der_proxy": eval_result.der_proxy,
                    "latency_s": latency,
                    "rtf": latency / (len(audio)/sr) if len(audio) > 0 else 0
                }
            })
            
        except Exception as e:
            logger.error(f"Error processing {case_id}: {e}")
            results.append({
                "id": case_id,
                "error": str(e),
                "metrics": None
            })

    # 4. Summary & Gates
    if not results:
        logger.error("No results produced")
        sys.exit(1)
        
    valid_results = [r for r in results if r.get("metrics")]
    
    # Compute total audio duration for run_context
    total_duration_s = sum(r.get("duration_s", 0) for r in valid_results)
    
    avg_metrics = {}
    if valid_results:
        avg_metrics = {
            "num_speakers_pred": float(np.mean([r["metrics"]["num_speakers_pred"] for r in valid_results])),
            "speaker_count_error": float(np.mean([r["metrics"]["speaker_count_error"] for r in valid_results])),
            "der_proxy": float(np.mean([r["metrics"]["der_proxy"] for r in valid_results])),
            "rtf": float(np.mean([r["metrics"]["rtf"] for r in valid_results]))
        }
    
    # Simple Gates for Smoke Test
    # For smoke, we just want to ensure it didn't fail catastrophically
    has_failure = any(r.get("error") for r in results)
    
    # Speaker Max Gate (e.g. shouldn't detect 10 speakers in a 1 speaker file)
    num_speakers_max = max([r["metrics"]["num_speakers_pred"] for r in valid_results]) if valid_results else 0
    speaker_gate_passed = num_speakers_max < 10 # Arbitrary safety
    
    summary = {
        "device": args.device,
        "total_duration_s": total_duration_s,  # For run_context
        "metrics": avg_metrics,
        "gates": {
            "has_failure": has_failure,
            "speaker_gate_passed": speaker_gate_passed,
            "num_speakers_max": num_speakers_max
        }
    }
    
    # Get dataset path for provenance
    dataset_path = Path(f"data/golden/{args.dataset}.yaml")
    # Determine if dataset has ground truth (expected speakers count)
    has_ground_truth = all("expected" in case for case in cases)
    
    # 5. Save Artifacts
    save_run_artifact(args.model, args.dataset, dataset_path, results, summary, Path(f"runs/{args.model}/diarization"), has_ground_truth)
    
    if has_failure:
        logger.warning(f"Run completed with failures")
        sys.exit(1)
    else:
        logger.info(f"Run completed successfully. Mean DER Proxy: {avg_metrics.get('der_proxy', 1.0):.2%}")
        sys.exit(0)

if __name__ == "__main__":
    main()
