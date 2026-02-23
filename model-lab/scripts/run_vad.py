#!/usr/bin/env python3
"""
VAD Runner Script
Run Voice Activity Detection inference and collect evidence.

Usage:
  python scripts/run_vad.py --model silero_vad --dataset vad_smoke_v1
  python scripts/run_vad.py --model silero_vad --audio file.wav  # Adhoc mode
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.media_ingest import FFmpegNotFoundError, IngestConfig, IngestError, ingest_media
from harness.metrics_vad import VADMetrics
from harness.preprocess_ops import results_to_artifact_section, run_preprocessing_chain
from harness.registry import load_model_from_config
from harness.run_provenance import create_provenance, create_run_context
from harness.runner_schema import (
    InputsSchema,
    QualityMetrics,
    RunContext,
    RunnerArtifact,
    validate_artifact,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def sha256_file(path: str) -> str:
    """Calculate SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_vad_adhoc(model_id: str, input_path: str, device: str = "cpu", pre: str = None):
    """
    Run VAD on a single media file (adhoc mode).

    Supports both audio and video files (mp4, mkv, etc).
    Produces artifact with grade=adhoc and structural metrics only.

    Args:
        pre: Comma-separated list of operators (e.g., "trim_silence,normalize_loudness")
    """
    input_path = Path(input_path).resolve()
    logger.info(f"=== VAD Adhoc: {model_id} on {input_path.name} ===")

    # Ingest media (handles video extraction if needed)
    try:
        ingest_artifacts_dir = Path(f"runs/{model_id}/vad/_adhoc_ingest")
        ingest = ingest_media(input_path, ingest_artifacts_dir, IngestConfig())
    except FFmpegNotFoundError as e:
        logger.error(f"❌ {e}")
        logger.error("   Video support requires ffmpeg. Install: brew install ffmpeg")
        sys.exit(4)
    except IngestError as e:
        logger.error(f"❌ Ingest failed: {e}")
        sys.exit(4)

    source_hash = ingest["source_media_hash"]
    audio_hash = ingest["audio_content_hash"]
    processed_audio_path = Path(ingest["processed_audio_path"])
    source_suffix = input_path.suffix.lower() or "<unknown>"
    is_extracted = processed_audio_path.resolve() != input_path.resolve()

    logger.info(f"Source: {source_hash[:12]} ({source_suffix})")
    logger.info(f"Audio: {audio_hash[:12]} ({ingest['duration_s']:.2f}s)")
    if is_extracted:
        logger.info(f"Extracted via: ffmpeg {ingest['ffmpeg_version']}")

    # Run preprocessing chain if specified
    preprocessing_results = []
    audio_for_model, sr_for_model = sf.read(str(processed_audio_path))
    if audio_for_model.ndim > 1:
        audio_for_model = audio_for_model.mean(axis=1)
    audio_for_model = audio_for_model.astype(np.float32)
    processed_duration_s = float(ingest["duration_s"])

    if pre:
        operators = [op.strip() for op in pre.split(",") if op.strip()]
        if operators:
            logger.info(f"Preprocessing: {' → '.join(operators)}")
            preprocessing_results = run_preprocessing_chain(
                audio_for_model, sr_for_model, operators
            )
            if preprocessing_results:
                final_result = preprocessing_results[-1]
                audio_for_model = final_result.audio
                sr_for_model = final_result.sample_rate
                processed_duration_s = final_result.duration_out_s
                logger.info(
                    f"After preprocessing: {final_result.out_audio_hash[:12]} ({processed_duration_s:.2f}s)"
                )

    try:
        # Load model
        config_path = Path(f"models/{model_id}/config.yaml")
        bundle = load_model_from_config(config_path, device=device)
        detector = bundle["vad"]["detect"]
        logger.info("✓ Model loaded")

        # Run VAD (on preprocessed audio)
        t0 = time.time()
        output = detector(audio_for_model, sr=sr_for_model)
        latency = time.time() - t0

        # RTF uses processed duration (what the model actually saw)
        rtf = latency / processed_duration_s if processed_duration_s > 0 else 0
        segments = output.get("segments", [])

        # Calculate metrics
        m_res = VADMetrics.calculate(segments, len(audio_for_model), sr_for_model)

        # Create schema-validated artifact
        schema_run_context = RunContext(
            task="vad",
            model_id=model_id,
            grade="adhoc",
            timestamp=datetime.now().isoformat(),
            git_hash=None,
            command=sys.argv,
            device=device,
        )

        # Hash of processed audio if preprocessing was applied
        final_audio_hash = (
            preprocessing_results[-1].out_audio_hash if preprocessing_results else audio_hash
        )

        schema_inputs = InputsSchema(
            audio_path=str(input_path),
            audio_hash=final_audio_hash,  # Hash of what model actually saw
            source_media_path=ingest.get("source_media_path"),
            source_media_hash=source_hash,
            dataset_id=f"adhoc_{audio_hash[:12]}",
            audio_duration_s=processed_duration_s,  # Duration of processed audio
            sample_rate=sr_for_model,
        )

        structural_metrics = {
            "rtf": rtf,
            "speech_ratio": m_res.speech_ratio,
            "num_segments": m_res.num_segments,
            "latency_ms": latency * 1000,
            "duration_s": processed_duration_s,
            "source_duration_s": float(ingest["duration_s"]),  # Original duration for reference
        }

        # Ingest provenance
        ingest_provenance = {
            "ingest_tool": "ffmpeg",
            "ingest_version": ingest["ffmpeg_version"],
            "is_extracted": is_extracted,
            "original_format": source_suffix,
        }

        schema_artifact = RunnerArtifact(
            run_context=schema_run_context,
            inputs=schema_inputs,
            metrics_quality=QualityMetrics(),  # All None for adhoc
            metrics_structural=structural_metrics,
            output={
                "segments_count": len(segments),
                "segments": segments[:20],
            },
            provenance={
                "has_ground_truth": False,
                "metrics_valid": True,
                **ingest_provenance,
            },
            gates={"has_failure": False},
        )

        # Validate before writing
        validate_artifact(schema_artifact)

        # Convert to dict and add preprocessing section
        result_obj = schema_artifact.to_dict()
        if preprocessing_results:
            result_obj["preprocessing"] = results_to_artifact_section(preprocessing_results)

        # Save artifact
        runs_dir = Path(f"runs/{model_id}/vad")
        runs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        run_file = runs_dir / f"adhoc_{timestamp}.json"

        with open(run_file, "w") as f:
            json.dump(result_obj, f, indent=2, default=str)

        logger.info("✓ Adhoc run completed")
        print(f"ARTIFACT_PATH:{run_file}")
        return result_obj, str(run_file)
    except Exception:
        raise


def main():
    parser = argparse.ArgumentParser(description="Run VAD Evaluation")
    parser.add_argument("--model", required=True, help="Model ID (e.g., silero_vad)")

    # Mutually exclusive: --dataset or --audio/--input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--dataset", help="Dataset ID (e.g., vad_smoke_v1)")
    input_group.add_argument("--audio", type=Path, help="Single audio/video file (adhoc mode)")
    input_group.add_argument("--input", type=Path, dest="audio", help="Alias for --audio")

    parser.add_argument("--device", default="cpu", help="Device to use (cpu, mps, cuda)")
    parser.add_argument(
        "--pre",
        type=str,
        default=None,
        help="Preprocessing operators (comma-separated, e.g., trim_silence,normalize_loudness)",
    )
    args = parser.parse_args()

    # Adhoc mode
    if args.audio:
        try:
            result, artifact_path = run_vad_adhoc(
                args.model, str(args.audio), args.device, pre=args.pre
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
    logger.info(f"Loading model: {args.model} on {args.device}")
    try:
        config_path = Path(f"models/{args.model}/config.yaml")
        bundle = load_model_from_config(config_path, device=args.device)
        detector = bundle["vad"]["detect"]
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
        dataset_id=dataset["id"],
        dataset_path=dataset_path,
        audio_path=None,  # Multiple audio files
        ground_truth_path=None,  # VAD smoke has no ground truth
        metrics_valid=True,
    )

    evidence_data = {
        "model_id": args.model,
        "job_id": run_id,
        "capability": "vad",  # Required for generate_decisions.py to detect task type
        "meta": {"task": "vad", "timestamp": timestamp},
        "input": {"dataset_id": dataset["id"], "case_count": len(dataset["cases"])},
        "system": {"device": args.device, "runtime": "python"},
        "manifest": {"timestamp": timestamp, "argv": sys.argv},
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
            "dataset_id": dataset["id"],
            "sanity_gates": {},
        },
    }

    metrics_accum = {
        "rtf": [],
        "speech_ratio": [],
        "num_segments": [],
        "duration_s": [],  # Track audio durations
    }

    failed_cases = 0

    for case in dataset["cases"]:
        audio_path = Path(case["audio_file"])
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}")
            continue

        case_id = case.get("id", audio_path.stem)
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
                "duration_s": duration_s,
            }

            metrics_accum["speech_ratio"].append(m_res.speech_ratio)
            metrics_accum["num_segments"].append(m_res.num_segments)
            metrics_accum["duration_s"].append(duration_s)

            case_result = {
                "id": case_id,
                "metrics": met,
                "gates": gates,
                "segments_count": len(raw_segments),
            }

            if gates["has_failure"]:
                failed_cases += 1
                logger.warning(f"Case {case_id} failed checks: {gates}")

            evidence_data["cases"].append(case_result)

        except Exception as e:
            logger.error(f"Failed case {case_id}: {e}")
            failed_cases += 1
            evidence_data["cases"].append(
                {"id": case_id, "error": str(e), "gates": {"has_failure": True}}
            )

    # 4. Computed Summary Metrics
    if metrics_accum["rtf"]:
        evidence_data["metrics"] = {
            "rtf": float(np.median(metrics_accum["rtf"])),
            "speech_ratio": float(np.mean(metrics_accum["speech_ratio"])),
            "num_segments": float(np.mean(metrics_accum["num_segments"])),
        }

    # Global Gates
    overall_failure = failed_cases > 0
    evidence_data["gates"] = {
        "has_failure": overall_failure,
        "failed_cases": failed_cases,
        "total_cases": len(dataset["cases"]),
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
