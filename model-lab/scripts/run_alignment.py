#!/usr/bin/env python3
"""
Run Alignment evaluation for a specific model and dataset.

Usage:
    uv run scripts/run_alignment.py --model whisper --dataset primary
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import yaml
import soundfile as sf
import numpy as np

# Add harness to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.registry import ModelRegistry
from harness.metrics_alignment import AlignmentMetrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_alignment")


def load_dataset(dataset_id: str) -> tuple[Dict[str, Any], Path]:
    # Reuse ASR datasets which have audio + text
    paths = [
        Path(f"data/golden/{dataset_id}.yaml"),
        Path(f"data/{dataset_id}.yaml")
    ]
    for p in paths:
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f), p
    raise FileNotFoundError(f"Dataset not found: {dataset_id}")


def save_run_artifact(model_id: str,
                      dataset_id: str,
                      results: List[Dict[str, Any]],
                      summary: Dict[str, Any],
                      output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    run_id = f"{dataset_id}_{int(time.time())}"
    
    artifact = {
        "meta": {
            "run_id": run_id,
            "timestamp": timestamp,
            "model_id": model_id,
            "task": "alignment"
        },
        "input": {
            "dataset_id": dataset_id,
            "count": len(results)
        },
        "system": {
            "device": summary.get("device", "unknown")
        },
        "metrics": summary.get("metrics", {}),
        "gates": summary.get("gates", {}),
        "evidence": {
            "grade": "golden_batch" if "primary" in dataset_id else "smoke",
            "dataset_id": dataset_id,
            "sanity_gates": summary.get("sanity_gates", {}),
            "wer_valid": True # Schema consistency
        },
        "results": results
    }
    
    outfile = output_dir / f"{run_id}.json"
    with open(outfile, "w") as f:
        json.dump(artifact, f, indent=2)
    
    logger.info(f"Saved run artifact: {outfile}")

from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Run Alignment evaluation")
    parser.add_argument("--model", required=True, help="Model ID")
    parser.add_argument("--dataset", required=True, help="Dataset ID (e.g., primary)")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda, mps)")
    
    args = parser.parse_args()
    
    # 1. Load Model
    try:
        config_path = Path(f"models/{args.model}/config.yaml")
        if not config_path.exists():
            raise ValueError(f"Config not found: {config_path}")
            
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        bundle = ModelRegistry.load_model(config['model_type'], config, device=args.device)
        
        if "alignment" not in bundle:
            raise ValueError(f"Model {args.model} validation failed: 'alignment' capability missing in bundle. Does loader implement it?")
            
        align_fn = bundle["alignment"]["align"]
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
        
    # 2. Load Dataset
    try:
        dataset, dataset_path = load_dataset(args.dataset)
        cases = dataset.get("cases") or dataset.get("test_cases", [])
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)
        
    logger.info(f"Starting Alignment run for {args.model} on {args.dataset} ({len(cases)} cases)")
    
    results = []
    
    for case in cases:
        case_id = case["id"]
        # Handle 'audio_file' vs 'input_audio' vs 'audio_path'
        audio_path_str = case.get("audio_file") or case.get("input_audio") or case.get("audio_path")
        
        if audio_path_str:
            # Try resolving relative to dataset location first
            p1 = (dataset_path.parent / audio_path_str).resolve()
            # Try resolving relative to CWD
            p2 = Path(audio_path_str).resolve()
            
            if p1.exists():
                audio_file = p1
            elif p2.exists():
                audio_file = p2
        
        if not audio_file or not audio_file.exists():
            logger.warning(f"Audio file missing: {audio_file}")
            continue
            
        try:
            # Load audio
            audio, sr = sf.read(str(audio_file))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            audio_duration = len(audio) / sr
            
            logger.info(f"Processing {case_id} ({audio_duration:.1f}s)...")
            
            t0 = time.time()
            output = align_fn(audio, sr=sr)
            latency = time.time() - t0
            
            segments = output.get("segments", [])
            
            # Compute Metrics
            eval_result = AlignmentMetrics.evaluate(
                segments=segments,
                audio_duration_s=audio_duration,
                latency_s=latency
            )
            
            # Simplified output storage
            results.append({
                "id": case_id,
                "audio_file": str(audio_file),
                "metrics": {
                    "num_segments": eval_result.num_segments,
                    "violations": eval_result.monotonicity_violations,
                    "neg_duration": eval_result.negative_duration_count,
                    "coverage": eval_result.coverage_ratio,
                    "latency_ms": latency * 1000
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
    if valid_results:
        avg_metrics = {
            "violations_mean": float(np.mean([r["metrics"]["violations"] for r in valid_results])),
            "coverage_mean": float(np.mean([r["metrics"]["coverage"] for r in valid_results]))
        }
        
    has_failure = any(r.get("error") for r in results)
    
    summary = {
        "device": args.device,
        "metrics": avg_metrics,
        "gates": {"has_failure": has_failure}
    }
    
    save_run_artifact(args.model, args.dataset, results, summary, Path(f"runs/{args.model}/alignment"))
    
    if has_failure:
        sys.exit(1)
    else:
        logger.info(f"Alignment Run completed. Mean Violations: {avg_metrics.get('violations_mean', 0):.2f}")
        sys.exit(0)

if __name__ == "__main__":
    main()