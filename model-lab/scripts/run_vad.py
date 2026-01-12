#!/usr/bin/env python3
"""
VAD Runner Script
Run Voice Activity Detection inference and collect evidence.

Usage:
  python scripts/run_vad.py --model silero_vad --dataset vad_smoke_v1
"""

import argparse
import json
import logging
import sys
import time
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.registry import ModelRegistry, load_model_from_config
from harness.metrics_vad import VADMetrics
from harness.run_provenance import create_provenance, create_run_context, RUN_SCHEMA_VERSION

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sha256_file(path: str) -> str:
    """Calculate SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Run VAD Evaluation")
    parser.add_argument("--model", required=True, help="Model ID (e.g., silero_vad)")
    parser.add_argument("--dataset", required=True, help="Dataset ID (e.g., vad_smoke_v1)")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu, mps, cuda)")
    args = parser.parse_args()

    # 1. Load Model
    logger.info(f"Loading model: {args.model} on {args.device}")
    try:
        config_path = Path(f"models/{args.model}/config.yaml")
        bundle = load_model_from_config(config_path, device=args.device)
        detector = bundle['vad']['detect']
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # 2. Load Dataset
    dataset_path = Path(f"data/golden/{args.dataset}.yaml")
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)
        
    with open(dataset_path) as f:
        dataset = yaml.safe_load(f)

    logger.info(f"Loaded dataset: {dataset['id']} ({len(dataset['cases'])} cases)")

    # 3. Run Inference
    results_dir = Path(f"runs/{args.model}/vad")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    # Unique run ID
    run_id = f"run_{int(time.time())}"
    run_file = results_dir / f"{run_id}.json"
    
    # Create provenance EARLY - before any metric computation
    provenance = create_provenance(
        dataset_id=dataset['id'],
        dataset_path=dataset_path,
        audio_path=None,  # Multiple audio files
        ground_truth_path=None,  # VAD smoke has no ground truth
        metrics_valid=True,
    )

    evidence_data = {
        "model_id": args.model,
        "job_id": run_id,
        "capability": "vad",  # Required for generate_decisions.py to detect task type
        "meta": {
            "task": "vad",
            "timestamp": timestamp
        },
        "input": {
            "dataset_id": dataset['id'],
            "case_count": len(dataset['cases'])
        },
        "system": {
            "device": args.device,
            "runtime": "python"
        },
        "manifest": {
            "timestamp": timestamp,
            "argv": sys.argv
        },
        "cases": [],
        "metrics": {},
        "gates": {},
        "provenance": provenance,  # REQUIRED for all runs
        "run_context": create_run_context(
            device=args.device,
            audio_duration_s=None,  # Computed after processing cases
        ),
        "evidence": {
             "grade": "smoke",  # Structural evidence - no ground truth but runner works
             "dataset_id": dataset['id'],
             "sanity_gates": {}
        }
    }

    metrics_accum = {
        "rtf": [],
        "speech_ratio": [],
        "num_segments": [],
        "duration_s": []  # Track audio durations
    }
    
    failed_cases = 0

    for case in dataset['cases']:
        audio_path = Path(case['audio_file'])
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}")
            continue
            
        case_id = case.get('id', audio_path.stem)
        logger.info(f"Processing case: {case_id}")

        # Read Audio
        try:
            audio, sr = sf.read(audio_path)
            # Ensure mono
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            # Ensure float32
            audio = audio.astype(np.float32)
            
            duration_s = len(audio) / sr
            
            # Inference
            t0 = time.time()
            output = detector(audio, sr=sr)
            t1 = time.time()
            
            infer_time = t1 - t0
            rtf = infer_time / duration_s
            
            metrics_accum["rtf"].append(rtf)
            
            # Retrieve segments
            raw_segments = output.get("segments", [])
            
            # Calculate metrics
            m_res = VADMetrics.calculate(raw_segments, len(audio), sr)
            
            # Check gates
            gates = VADMetrics.check_gates(m_res, case)
            
            met = {
                "rtf": rtf,
                "speech_ratio": m_res.speech_ratio,
                "num_segments": m_res.num_segments,
                "duration_s": duration_s
            }
            
            metrics_accum["speech_ratio"].append(m_res.speech_ratio)
            metrics_accum["num_segments"].append(m_res.num_segments)
            metrics_accum["duration_s"].append(duration_s)

            case_result = {
                "id": case_id,
                "metrics": met,
                "gates": gates,
                "segments_count": len(raw_segments)
            }
            
            if gates["has_failure"]:
                failed_cases += 1
                logger.warning(f"Case {case_id} failed checks: {gates}")
            
            evidence_data["cases"].append(case_result)
            
        except Exception as e:
            logger.error(f"Failed case {case_id}: {e}")
            failed_cases += 1
            evidence_data["cases"].append({
                "id": case_id,
                "error": str(e),
                "gates": {"has_failure": True}
            })

    # 4. Computed Summary Metrics
    if metrics_accum["rtf"]:
        evidence_data["metrics"] = {
            "rtf": float(np.median(metrics_accum["rtf"])),
            "speech_ratio": float(np.mean(metrics_accum["speech_ratio"])),
            "num_segments": float(np.mean(metrics_accum["num_segments"]))
        }
    
    # Global Gates
    overall_failure = failed_cases > 0
    evidence_data["gates"] = {
        "has_failure": overall_failure,
        "failed_cases": failed_cases,
        "total_cases": len(dataset['cases'])
    }

    # Update run_context with computed audio duration
    total_duration = sum(metrics_accum["duration_s"]) if metrics_accum["duration_s"] else None
    evidence_data["run_context"]["audio_duration_s"] = total_duration

    # Write output
    with open(run_file, "w") as f:
        json.dump(evidence_data, f, indent=2)
        
    logger.info(f"Run complete. Saved to {run_file}")
    
    if overall_failure:
        logger.warning("Run checks FAILED")
        sys.exit(1)
    else:
        logger.info("Run checks PASSED")

if __name__ == "__main__":
    main()
