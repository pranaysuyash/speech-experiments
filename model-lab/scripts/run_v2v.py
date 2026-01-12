#!/usr/bin/env python3
"""
Run V2V evaluation for a specific model and dataset.

Usage:
    uv run scripts/run_v2v.py --model lfm2_5_audio --dataset v2v_smoke_v1
    uv run scripts/run_v2v.py --model lfm2_5_audio --audio file.wav --prompt "Hello"  # Adhoc
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
from harness.metrics_v2v import V2VMetrics
from harness.run_provenance import create_provenance, create_run_context, RUN_SCHEMA_VERSION
from harness.adhoc_dataset import create_adhoc_dataset
from harness.runner_schema import (
    RunnerArtifact, RunContext, InputsSchema, QualityMetrics,
    validate_artifact, compute_pcm_hash,
)
from harness.media_ingest import ingest_media, IngestError, FFmpegNotFoundError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_v2v")


def load_dataset(dataset_id: str) -> Dict[str, Any]:
    dataset_path = Path(f"data/golden/{dataset_id}.yaml")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with open(dataset_path) as f:
        return yaml.safe_load(f)


def run_v2v_adhoc(model_id: str, input_path: str, prompt: str = None, device: str = "cpu"):
    """
    Run V2V on a single media file (adhoc mode).
    
    Supports both audio and video files (mp4, mkv, etc).
    Produces artifact with grade=adhoc and structural metrics only.
    """
    input_path = Path(input_path).resolve()
    logger.info(f"=== V2V Adhoc: {model_id} on {input_path.name} ===")
    if prompt:
        logger.info(f"Prompt: {prompt[:50]}...")
    
    # Ingest media (handles video extraction if needed)
    try:
        ingest = ingest_media(input_path)
    except FFmpegNotFoundError as e:
        logger.error(f"❌ {e}")
        logger.error("   Video support requires ffmpeg. Install: brew install ffmpeg")
        sys.exit(4)
    except IngestError as e:
        logger.error(f"❌ Ingest failed: {e}")
        sys.exit(4)
    
    logger.info(f"Source: {ingest.source_media_hash[:12]} ({ingest.original_format})")
    logger.info(f"Audio: {ingest.audio_hash[:12]} ({ingest.audio_duration_s:.2f}s)")
    if ingest.is_extracted:
        logger.info(f"Extracted via: {ingest.ingest_tool} {ingest.ingest_version}")
    
    try:
        # Load model
        config_path = Path(f"models/{model_id}/config.yaml")
        if not config_path.exists():
            raise ValueError(f"Config not found: {config_path}")
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        bundle = ModelRegistry.load_model(config['model_type'], config, device=device)
        
        if "v2v" not in bundle:
            raise ValueError(f"Model {model_id} does not support 'v2v'")
        
        v2v_fn = bundle["v2v"]["run_v2v_turn"]
        logger.info(f"✓ Model loaded")
        
        # Run V2V
        t0 = time.time()
        output = v2v_fn(audio=ingest.audio, prompt=prompt) if prompt else v2v_fn(audio=ingest.audio)
        latency = time.time() - t0
        
        # Get output
        response_audio = output.get("audio")
        response_text = output.get("response_text", "")
        sr_out = output.get("sample_rate", 24000)
        
        # Compute metrics
        eval_result = V2VMetrics.evaluate(
            response_audio=response_audio,
            sr=sr_out,
            latency_s=latency
        )
        
        rtf_like = latency / ingest.audio_duration_s if ingest.audio_duration_s > 0 else None
        
        # Create schema-validated artifact
        schema_run_context = RunContext(
            task="v2v",
            model_id=model_id,
            grade="adhoc",
            timestamp=datetime.now().isoformat(),
            git_hash=None,
            command=sys.argv,
            device=device,
        )
        
        schema_inputs = InputsSchema(
            audio_path=str(input_path),
            audio_hash=ingest.audio_hash,
            source_media_path=str(ingest.source_media_path) if ingest.is_extracted else None,
            source_media_hash=ingest.source_media_hash if ingest.is_extracted else None,
            dataset_id=f"adhoc_{ingest.audio_hash[:12]}",
            audio_duration_s=ingest.audio_duration_s,
            sample_rate=ingest.sample_rate,
        )
        
        structural_metrics = {
            'latency_ms': eval_result.latency_ms,
            'input_duration_s': ingest.audio_duration_s,
            'response_duration_s': eval_result.response_duration_s,
            'rtf_like': rtf_like,
            'has_audio': eval_result.has_audio,
        }
        
        # Ingest provenance
        ingest_provenance = {
            'ingest_tool': ingest.ingest_tool,
            'ingest_version': ingest.ingest_version,
            'is_extracted': ingest.is_extracted,
            'original_format': ingest.original_format,
        }
        
        schema_artifact = RunnerArtifact(
            run_context=schema_run_context,
            inputs=schema_inputs,
            metrics_quality=QualityMetrics(),
            metrics_structural=structural_metrics,
            output={
                'text': response_text,
                'has_audio': eval_result.has_audio,
                'duration_s': eval_result.response_duration_s,
                'prompt': prompt,
            },
            provenance={
                'has_ground_truth': False,
                'metrics_valid': True,
                **ingest_provenance,
            },
            gates={'has_failure': not eval_result.has_audio},
        )
        
        # Validate before writing
        validate_artifact(schema_artifact)
        
        # Save artifact
        runs_dir = Path(f'runs/{model_id}/v2v')
        runs_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        run_file = runs_dir / f'adhoc_{timestamp}.json'
        
        with open(run_file, 'w') as f:
            json.dump(schema_artifact.to_dict(), f, indent=2, default=str)
        
        logger.info(f"✓ Adhoc run completed: response {eval_result.response_duration_s:.1f}s")
        print(f"ARTIFACT_PATH:{run_file}")
        return schema_artifact.to_dict(), str(run_file)
        
    finally:
        ingest.cleanup()


def save_run_artifact(model_id: str,
                      dataset_id: str,
                      dataset_path: Path,
                      results: List[Dict[str, Any]],
                      summary: Dict[str, Any],
                      output_dir: Path):
    """Save V2V run artifact with provenance and run_context.
    
    V2V smoke evidence: has_ground_truth=false (no reference target).
    Structural metrics (latency, rtf) are valid.
    Quality metrics (WER, accuracy) must be None.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    run_id = f"{dataset_id}_{int(time.time())}"
    
    # Create provenance EARLY - V2V smoke has NO ground truth
    provenance = create_provenance(
        dataset_id=dataset_id,
        dataset_path=dataset_path,
        audio_path=None,  # Multiple audio files
        ground_truth_path=None,  # V2V smoke has no reference target
        metrics_valid=True,  # Structural metrics are valid
    )
    
    # Create run_context for interpretable latency
    run_context = create_run_context(
        device=summary.get("device", "unknown"),
        audio_duration_s=summary.get("metrics", {}).get("input_duration_s"),
        model_version=None,  # TODO: Get from config
    )
    
    artifact = {
        "meta": {
            "run_id": run_id,
            "timestamp": timestamp,
            "model_id": model_id,
            "task": "v2v"
        },
        "capability": "v2v",  # Required for task detection
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
        "run_context": run_context,  # REQUIRED for interpretable latency
        "evidence": {
            "grade": "smoke" if "smoke" in dataset_id else "golden_batch",
            "dataset_id": dataset_id,
            "sanity_gates": summary.get("sanity_gates", {})
        },
        "results": results
    }
    
    outfile = output_dir / f"{run_id}.json"
    with open(outfile, "w") as f:
        json.dump(artifact, f, indent=2)
    
    logger.info(f"Saved run artifact: {outfile}")


def main():
    parser = argparse.ArgumentParser(description="Run V2V evaluation")
    parser.add_argument("--model", required=True, help="Model ID")
    
    # Mutually exclusive: --dataset or --audio/--input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--dataset", help="Dataset ID")
    input_group.add_argument("--audio", type=Path, help="Single audio/video file (adhoc mode)")
    input_group.add_argument("--input", type=Path, dest='audio', help="Alias for --audio")
    
    parser.add_argument("--prompt", help="Text prompt for V2V (adhoc mode only)")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda, mps)")
    
    args = parser.parse_args()
    
    # Adhoc mode
    if args.audio:
        try:
            result, artifact_path = run_v2v_adhoc(
                args.model, str(args.audio), 
                prompt=args.prompt, device=args.device
            )
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
            raise ValueError(f"Config not found: {config_path}")
            
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        bundle = ModelRegistry.load_model(config['model_type'], config, device=args.device)
        
        if "v2v" not in bundle:
            raise ValueError(f"Model {args.model} does not support 'v2v' capability")
            
        v2v_fn = bundle["v2v"]["run_v2v_turn"]
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Save a failure artifact?
        sys.exit(1)
        
    # 2. Load Dataset
    try:
        dataset = load_dataset(args.dataset)
        cases = dataset.get("cases", [])
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)
        
    logger.info(f"Starting V2V run for {args.model} on {args.dataset} ({len(cases)} cases)")
    
    results = []
    
    for case in cases:
        case_id = case["id"]
        input_audio_path = Path(case["input_audio"])
        expected = case.get("expected", {})
        
        if not input_audio_path.exists():
            logger.warning(f"Input audio missing: {input_audio_path}")
            continue
            
        try:
            # Load input audio
            audio, sr = sf.read(str(input_audio_path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            
            logger.info(f"Processing {case_id} ({len(audio)/sr:.1f}s audio)...")
            
            t0 = time.time()
            output = v2v_fn(audio=audio) 
            latency = time.time() - t0
            
            # Retrieve outputs
            response_audio = output.get("audio")
            response_text = output.get("response_text", "")
            sr_out = output.get("sample_rate", 24000)
            
            # Compute Metrics
            eval_result = V2VMetrics.evaluate(
                response_audio=response_audio,
                sr=sr_out,
                latency_s=latency
            )
            
            input_duration_s = len(audio) / sr
            rtf_like = (latency * 1000) / (input_duration_s * 1000) if input_duration_s > 0 else None
            
            results.append({
                "id": case_id,
                "input_audio": str(input_audio_path),
                "input_duration_s": input_duration_s,
                "output": {
                    "text": response_text,
                    "has_audio": eval_result.has_audio,
                    "duration_s": eval_result.response_duration_s
                },
                "metrics": {
                    "latency_ms": eval_result.latency_ms,
                    "input_duration_s": input_duration_s,
                    "response_duration_s": eval_result.response_duration_s,
                    "rtf_like": rtf_like,  # latency/input_duration - for stable ranking
                    "rtf": latency / eval_result.response_duration_s if eval_result.response_duration_s > 0 else 0
                }
            })
            
        except Exception as e:
            logger.error(f"Error processing {case_id}: {e}")
            results.append({
                "id": case_id,
                "error": str(e),
                "metrics": None
            })
            
    # Summary
    if not results:
        logger.error("No results produced")
        sys.exit(1)
        
    valid_results = [r for r in results if r.get("metrics")]
    avg_metrics = {}
    total_input_duration = 0.0
    if valid_results:
        total_input_duration = sum(r.get("input_duration_s", 0) for r in valid_results)
        avg_rtf_like = np.mean([r["metrics"]["rtf_like"] for r in valid_results if r["metrics"].get("rtf_like") is not None])
        avg_metrics = {
            "latency_ms": float(np.mean([r["metrics"]["latency_ms"] for r in valid_results])),
            "input_duration_s": total_input_duration,
            "response_duration_s": float(np.mean([r["metrics"]["response_duration_s"] for r in valid_results])),
            "rtf_like": float(avg_rtf_like) if not np.isnan(avg_rtf_like) else None,
        }
        
    has_failure = any(r.get("error") for r in results)
    
    summary = {
        "device": args.device,
        "metrics": avg_metrics,
        "gates": {"has_failure": has_failure}
    }
    
    # Get dataset path for provenance
    dataset_path = Path(f"data/golden/{args.dataset}.yaml")
    
    save_run_artifact(args.model, args.dataset, dataset_path, results, summary, Path(f"runs/{args.model}/v2v"))
    
    if has_failure:
        sys.exit(1)
    else:
        logger.info("V2V Run completed successfully")
        sys.exit(0)

if __name__ == "__main__":
    main()
