#!/usr/bin/env python3
"""
Headless ASR runner for production testing.
Uses Bundle Contract v1 - no per-model special casing.

Usage: uv run python -m scripts.run_asr --model faster_whisper --dataset primary
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add harness to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from harness.env import load_dotenv_if_present

from harness.audio_io import AudioLoader, GroundTruthLoader
from harness.media_ingest import FFmpegNotFoundError, IngestConfig, IngestError, ingest_media
from harness.metrics_asr import ASRMetrics, diagnose_output_quality
from harness.preprocess_ops import results_to_artifact_section, run_preprocessing_chain
from harness.protocol import NormalizationValidator, RunContract, create_validation_report
from harness.registry import ModelRegistry
from harness.run_provenance import (
    can_compute_quality_metrics,
    create_provenance,
    create_run_context,
)
from harness.runner_schema import (
    InputsSchema,
    QualityMetrics,
    RunContext,
    RunnerArtifact,
    validate_artifact,
)
from harness.timers import PerformanceTimer


def compute_file_sha256(path: Path) -> str:
    """Compute SHA256 hash of a file for pairing integrity."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]  # First 16 chars for brevity


def compute_text_sha256(text: str) -> str:
    """Compute SHA256 hash of text content."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def determine_evidence_grade(dataset: str, has_ground_truth: bool) -> str:
    """
    Determine evidence grade for the run.

    golden_batch: from harness with ground truth, matched dataset
    smoke: quick validation run (structural only, no quality metrics)
    adhoc: manual/script run without proper pairing
    """
    # Golden datasets (with ground truth)
    golden_datasets = {"llm_primary", "primary", "asr_golden_v1", "ux_primary"}
    if dataset in golden_datasets and has_ground_truth:
        return "golden_batch"

    # Smoke datasets (structural only - may still have ground truth file but we don't compute quality)
    if (
        dataset.startswith("smoke")
        or dataset.endswith("_smoke_v1")
        or dataset.startswith("asr_smoke")
    ):
        return "smoke"

    return "adhoc"


def compute_sanity_gates(hyp_text: str, ref_text: str, duration_s: float) -> dict:
    """
    Compute sanity gates to flag potentially invalid WER.

    Returns dict with gate results and overall wer_valid.
    """
    gates = {"wer_valid": True, "invalid_reasons": []}

    if not ref_text or not hyp_text:
        gates["wer_valid"] = False
        gates["invalid_reasons"].append("missing_text")
        return gates

    hyp_words = len(hyp_text.split())
    ref_words = len(ref_text.split())

    # Length ratio gate (flag if hyp is >3x or <0.33x ref)
    if ref_words > 0:
        length_ratio = hyp_words / ref_words
        gates["length_ratio"] = length_ratio
        if length_ratio > 3.0 or length_ratio < 0.33:
            gates["wer_valid"] = False
            gates["invalid_reasons"].append(f"extreme_length_ratio:{length_ratio:.2f}")

    # Words-per-second gate (flag if > 5 wps or < 0.5 wps for long audio)
    if duration_s > 10:
        ref_wps = ref_words / duration_s
        hyp_wps = hyp_words / duration_s
        gates["ref_wps"] = ref_wps
        gates["hyp_wps"] = hyp_wps
        if ref_wps > 5 or ref_wps < 0.5:
            gates["wer_valid"] = False
            gates["invalid_reasons"].append(f"absurd_ref_wps:{ref_wps:.2f}")
        if hyp_wps > 5:
            gates["wer_valid"] = False
            gates["invalid_reasons"].append(f"absurd_hyp_wps:{hyp_wps:.2f}")

    return gates


def load_model_config(model_id: str) -> dict:
    """Load model configuration."""
    # Check models directory for config
    config_path = Path(f"models/{model_id}/config.yaml")

    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
            config.setdefault("model_type", model_id)
            return config

    # Fallback: minimal config for models without config files
    return {"model_type": model_id, "model_name": model_id, "audio": {"sample_rate": 16000}}


def get_dataset_files(dataset: str) -> tuple:
    """Get audio and ground truth files for dataset."""
    datasets = {
        "smoke": {
            "audio": Path("data/audio/SMOKE/conversation_2ppl_10s.wav"),
            "text": None,  # No ground truth - structural sanity only (provenance unclear)
        },
        "asr_smoke_v1": {
            "audio": Path("data/audio/PRIMARY/llm_recording_pranay.wav"),
            "text": Path("data/text/PRIMARY/llm.txt"),
        },
        "primary": {
            "audio": Path("data/audio/PRIMARY/llm_recording_pranay.wav"),
            "text": Path("data/text/PRIMARY/llm.txt"),
        },
        "llm_primary": {
            "audio": Path("data/audio/PRIMARY/llm_recording_pranay.wav"),
            "text": Path("data/text/PRIMARY/llm.txt"),
        },
        "conversation": {
            "audio": Path("data/audio/PRIMARY/UX_Psychology_From_Miller_s_Law_to_AI.wav"),
            "text": None,  # No ground truth for conversation
        },
        # Second golden dataset for min_runs=2
        "ux_primary": {
            "audio": Path("data/audio/PRIMARY/ux_psychology_30s.wav"),
            "text": Path("data/text/PRIMARY/ux_psychology_30s.txt"),
        },
    }

    if dataset not in datasets:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(datasets.keys())}")

    return datasets[dataset]["audio"], datasets[dataset]["text"]


def run_asr_test(model_id: str, dataset: str, device: str = None):
    """
    Run ASR test using Bundle Contract v1.

    No per-model transcription functions - all models accessed via bundle["asr"]["transcribe"]().
    """
    print(f"=== ASR Test: {model_id} on {dataset} ===")

    # Load model config
    config = load_model_config(model_id)
    model_name = config.get("model_name", model_id)
    print(f"Model: {model_name}")

    # Determine device
    if device is None:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model via registry - returns Bundle Contract v1
    bundle = ModelRegistry.load_model(model_id, config, device)
    print(f"✓ Model loaded (capabilities: {bundle['capabilities']})")

    # Verify ASR capability exists
    if "asr" not in bundle["capabilities"]:
        raise ValueError(f"Model {model_id} does not have ASR capability")

    # Get the transcribe function from bundle - this is the ONLY way to call the model
    transcribe_fn = bundle["asr"]["transcribe"]

    # Load dataset
    audio_path, text_path = get_dataset_files(dataset)
    print(f"Audio: {audio_path}")

    # Create provenance EARLY - before any metric computation
    provenance = create_provenance(
        dataset_id=dataset,
        dataset_path=None,  # TODO: Add dataset definition file
        audio_path=audio_path,
        ground_truth_path=text_path,
        metrics_valid=True,
    )
    print(f"Provenance: has_ground_truth={provenance['has_ground_truth']}")

    sample_rate = config.get("audio", {}).get("sample_rate", 16000)
    loader = AudioLoader(target_sample_rate=sample_rate)
    audio, sr, metadata = loader.load_audio(audio_path, model_id)

    # Load ground truth if available
    ground_truth = None
    if text_path and text_path.exists():
        ground_truth = GroundTruthLoader.load_text(text_path)
        print(f"Ground truth: {len(ground_truth)} chars")

    # Create run manifest
    manifest = RunContract.create_run_manifest(
        model_id, Path(f"models/{model_id}/config.yaml"), audio_path, text_path
    )

    # Transcribe using bundle contract - single line, works for ALL models
    print("Transcribing...")
    timer = PerformanceTimer()

    with timer.time_operation(f"{model_id}_transcribe") as timing_container:
        # THIS IS THE KEY: one call surface for all models
        result = transcribe_fn(audio, sr=sr, language="en")
        text = (result.get("text") or "").strip()
        segments = result.get("segments", [])

    timing = timing_container["result"]
    latency_ms = timing.elapsed_time_ms
    print(f"✓ Transcription: {len(text)} chars in {latency_ms:.1f}ms")

    # Apply normalization protocol
    normalized_ref = None
    normalized_hyp = None
    if ground_truth:
        normalized_ref = NormalizationValidator.normalize_text(ground_truth)
        normalized_hyp = NormalizationValidator.normalize_text(text)
        print(
            f"✓ Normalization applied (protocol v{NormalizationValidator.NORMALIZATION_PROTOCOL['version']})"
        )

    # Compute evidence integrity fields
    audio_sha256 = compute_file_sha256(audio_path)
    truth_sha256 = compute_text_sha256(ground_truth) if ground_truth else None
    evidence_grade = determine_evidence_grade(dataset, ground_truth is not None)

    # Compute sanity gates if we have ground truth
    sanity_gates = None
    if ground_truth and normalized_hyp:
        sanity_gates = compute_sanity_gates(
            normalized_hyp, normalized_ref, metadata["duration_seconds"]
        )
        if not sanity_gates["wer_valid"]:
            print(f"⚠️  Sanity gates failed: {sanity_gates['invalid_reasons']}")

    # Format gates for Schema 2.0 (String values only)
    gates = {}
    if sanity_gates:
        if sanity_gates["wer_valid"]:
            gates["wer_valid"] = "✅ Pass"
        else:
            reasons = "; ".join(sanity_gates["invalid_reasons"])
            gates["wer_valid"] = f"❌ {reasons}"

        # Add other metrics as info (optional, or just rely on wer_valid)
        if "length_ratio" in sanity_gates:
            gates["length_chk"] = f"ℹ️ {sanity_gates['length_ratio']:.2f}"

    # Build result object
    result_obj = {
        "provider_id": model_id,
        "capability": "asr",
        "input": {
            "audio_file": str(audio_path.name),
            "audio_sha256": audio_sha256,
            "duration_s": metadata["duration_seconds"],
            "sr": sr,
        },
        "output": {
            "text": text,
            "text_length": len(text),
            "normalized_text": normalized_hyp,
            "segments": segments[:10] if segments else [],  # First 10 segments only
        },
        "metrics": {
            "latency_ms_p50": latency_ms,
            "rtf": latency_ms / 1000 / metadata["duration_seconds"],
        },
        "system": {
            "device": bundle["device"],
            "model": model_name,
            "inference_type": "local",
            "capabilities": bundle["capabilities"],
        },
        "evidence": {
            "grade": evidence_grade,
            "dataset_id": dataset,
            "truth_sha256": truth_sha256,
            "sanity_gates": gates,  # Use normalized gates
            "wer_valid": sanity_gates["wer_valid"] if sanity_gates else None,
        },
        "protocol": {
            "normalization_version": NormalizationValidator.NORMALIZATION_PROTOCOL["version"],
            "bundle_contract": "v1",
        },
        "manifest": manifest,
        "timestamps": {
            "started_at": datetime.now().isoformat(),
            "finished_at": datetime.now().isoformat(),
        },
        "errors": [],
    }

    # Add provenance to result - REQUIRED for all runs
    result_obj["provenance"] = provenance

    # Add run_context for interpretable latency metrics
    result_obj["run_context"] = create_run_context(
        device=bundle["device"],
        audio_duration_s=metadata["duration_seconds"],
    )

    # Calculate quality metrics ONLY if:
    # 1. Ground truth exists
    # 2. Provenance allows quality metrics
    # 3. Grade is NOT smoke (smoke is structural only)
    is_smoke = evidence_grade == "smoke"

    if ground_truth and can_compute_quality_metrics(provenance) and not is_smoke:
        # ASR metrics
        asr_metrics = ASRMetrics.evaluate(
            transcription=normalized_hyp,
            ground_truth=normalized_ref,
            audio_duration_s=metadata["duration_seconds"],
            latency_s=latency_ms / 1000,
        )

        result_obj["metrics"]["wer"] = asr_metrics.wer
        result_obj["metrics"]["cer"] = asr_metrics.cer
        result_obj["output"]["ground_truth"] = ground_truth
        result_obj["output"]["ground_truth_normalized"] = normalized_ref
        result_obj["output"]["ground_truth_length"] = len(ground_truth)

        # Output quality diagnosis (truncation/hallucination/repetition)
        diagnosis = diagnose_output_quality(normalized_ref, normalized_hyp)
        result_obj["output_quality"] = diagnosis

        if diagnosis["has_failure"]:
            result_obj["errors"].append(f"Output quality issue: {diagnosis}")

        # Validation report
        validation = create_validation_report(model_id, ground_truth, text)
        result_obj["validation"] = validation

        print(f"WER: {asr_metrics.wer:.3f} ({asr_metrics.wer * 100:.1f}%)")
        print(f"CER: {asr_metrics.cer:.3f} ({asr_metrics.cer * 100:.1f}%)")
        print(f"RTF: {asr_metrics.rtv:.3f}x")

        # Show diagnosis warnings
        if diagnosis["is_truncated"]:
            print(f"⚠️  TRUNCATED: length_ratio={diagnosis['length_ratio']:.2f}")
        if diagnosis["is_hallucinating"]:
            print(f"⚠️  HALLUCINATING: length_ratio={diagnosis['length_ratio']:.2f}")
        if diagnosis["is_repetitive"]:
            print(
                f"⚠️  REPETITIVE: unique={diagnosis['unique_token_ratio']:.2f}, 3gram={diagnosis['repeat_3gram_rate']:.2f}"
            )
    else:
        # No ground truth = no quality metrics. This is intentional.
        result_obj["metrics"]["wer"] = None
        result_obj["metrics"]["cer"] = None
        print("ℹ️  No ground truth - WER/CER set to None (structural run only)")

    # Save results
    results_dir = Path(f"runs/{model_id}/asr")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Timestamped file (local only)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_file = results_dir / f"{timestamp}.json"
    with open(result_file, "w") as f:
        json.dump(result_obj, f, indent=2)

    # Summary file (tracked in git) - overwrite
    summary_file = results_dir / "summary.json"
    summary = {
        "model_id": model_id,
        "dataset": dataset,
        "last_run": datetime.now().isoformat(),
        "wer": result_obj["metrics"].get("wer"),
        "cer": result_obj["metrics"].get("cer"),
        "rtf": result_obj["metrics"]["rtf"],
        "latency_ms": latency_ms,
        "has_output_quality_failure": result_obj.get("output_quality", {}).get(
            "has_failure", False
        ),
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Results saved to: {result_file}")
    print(f"✓ Summary updated: {summary_file}")
    return result_obj


def run_asr_adhoc(model_id: str, input_path: str, device: str = None, pre: str = None):
    """
    Run ASR on a single media file (adhoc mode).

    Supports both audio and video files (mp4, mkv, etc).
    Produces artifact with:
    - grade = adhoc
    - has_ground_truth = false
    - structural metrics only (no WER)
    - source_media_hash for video containers
    - preprocessing chain if --pre specified

    Args:
        pre: Comma-separated list of operators (e.g., "trim_silence,normalize_loudness")
    """
    input_path = Path(input_path).resolve()
    print(f"=== ASR Adhoc: {model_id} on {input_path.name} ===")

    # Ingest media (handles audio/video through ffmpeg canonicalization)
    try:
        ingest_cfg = IngestConfig()
        ingest_artifacts_dir = Path(f"runs/{model_id}/asr/_adhoc_ingest")
        ingest = ingest_media(input_path, ingest_artifacts_dir, ingest_cfg)
    except FFmpegNotFoundError as e:
        print(f"❌ {e}")
        print("   Video support requires ffmpeg. Install: brew install ffmpeg")
        sys.exit(4)  # Runner error
    except IngestError as e:
        print(f"❌ Ingest failed: {e}")
        sys.exit(4)

    source_hash = ingest["source_media_hash"]
    audio_hash = ingest["audio_content_hash"]
    duration_s = float(ingest["duration_s"])
    processed_audio_path = Path(ingest["processed_audio_path"])
    source_suffix = input_path.suffix.lower() or "<unknown>"
    print(f"Source: {source_hash[:12]} ({source_suffix})")
    print(f"Audio: {audio_hash[:12]} ({duration_s:.2f}s)")
    print(f"Ingest: ffmpeg ({ingest['ffmpeg_version']})")

    # Run preprocessing chain if specified
    preprocessing_results = []
    sample_rate = load_model_config(model_id).get("audio", {}).get("sample_rate", 16000)
    audio_loader = AudioLoader(target_sample_rate=sample_rate)
    audio_for_model, sr_for_model, _ = audio_loader.load_audio(processed_audio_path, model_id)

    if pre:
        operators = [op.strip() for op in pre.split(",") if op.strip()]
        if operators:
            print(f"Preprocessing: {' → '.join(operators)}")
            preprocessing_results = run_preprocessing_chain(
                audio_for_model, sr_for_model, operators
            )
            if preprocessing_results:
                final_result = preprocessing_results[-1]
                audio_for_model = final_result.audio
                sr_for_model = final_result.sample_rate
                print(
                    f"After preprocessing: {final_result.out_audio_hash[:12]} ({final_result.duration_out_s:.2f}s)"
                )

    try:
        # Load model config
        config = load_model_config(model_id)
        model_name = config.get("model_name", model_id)
        print(f"Model: {model_name}")

        # Determine device
        if device is None:
            import torch

            device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Device: {device}")

        # Load model via registry
        bundle = ModelRegistry.load_model(model_id, config, device)
        print(f"✓ Model loaded (capabilities: {bundle['capabilities']})")

        if "asr" not in bundle["capabilities"]:
            raise ValueError(f"Model {model_id} does not have ASR capability")

        transcribe_fn = bundle["asr"]["transcribe"]

        # Run transcription with timing (on preprocessed audio)
        start_time = time.time()
        result = transcribe_fn(audio_for_model, sr_for_model)
        elapsed_s = time.time() - start_time

        # Get transcript
        if isinstance(result, dict):
            transcript = result.get("text", "")
            segments = result.get("segments", [])
        else:
            transcript = str(result)
            segments = []

        print("\n--- Transcript ---")
        print(transcript[:200] + "..." if len(transcript) > 200 else transcript)

        # Compute STRUCTURAL metrics only (no quality metrics for adhoc)
        rtf = elapsed_s / duration_s if duration_s > 0 else 0

        # Create schema-validated artifact
        schema_run_context = RunContext(
            task="asr",
            model_id=model_id,
            grade="adhoc",
            timestamp=datetime.now().isoformat(),
            git_hash=None,  # Will be populated
            command=sys.argv,
            device=device,
            model_version=config.get("model_version"),
        )

        # Track source_media_hash for video containers
        schema_inputs = InputsSchema(
            audio_path=str(input_path),
            audio_hash=audio_hash,  # Hash of decoded PCM
            source_media_path=str(input_path),
            source_media_hash=source_hash,
            dataset_id=f"adhoc_{audio_hash[:12]}",
            dataset_hash=None,
            audio_duration_s=duration_s,
            sample_rate=sr_for_model,
        )

        # Quality metrics MUST be None for adhoc - schema enforces this
        schema_quality = QualityMetrics()  # All None

        structural_metrics = {
            "latency_ms": elapsed_s * 1000,
            "duration_s": duration_s,
            "rtf": rtf,
            "word_count": len(transcript.split()) if transcript else 0,
            "segment_count": len(segments),
        }

        # Ingest metadata for provenance
        ingest_provenance = {
            "ingest_tool": "ffmpeg",
            "ingest_version": ingest["ffmpeg_version"],
            "is_extracted": processed_audio_path.resolve() != input_path.resolve(),
            "original_format": source_suffix,
            "processed_audio_path": str(processed_audio_path),
        }

        schema_artifact = RunnerArtifact(
            run_context=schema_run_context,
            inputs=schema_inputs,
            metrics_quality=schema_quality,
            metrics_structural=structural_metrics,
            output={
                "text": transcript,
                "segments": segments[:10] if segments else [],  # Limit stored
            },
            artifacts={},
            provenance={
                "has_ground_truth": False,
                "metrics_valid": True,
                **ingest_provenance,
            },
            gates={},
            errors=[],
        )

        # Validate artifact before writing - raises if contract violated
        validate_artifact(schema_artifact)

        # Convert to dict for JSON serialization
        result_obj = schema_artifact.to_dict()

        # Add preprocessing section if operators were run
        if preprocessing_results:
            result_obj["preprocessing"] = results_to_artifact_section(preprocessing_results)

        # Save artifact
        runs_dir = Path(f"runs/{model_id}/asr")
        runs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        result_file = runs_dir / f"adhoc_{timestamp}.json"

        with open(result_file, "w") as f:
            json.dump(result_obj, f, indent=2, default=str)

        # Print artifact path as LAST LINE for model_app parsing
        print("\n✓ Adhoc run completed successfully")
        print(f"ARTIFACT_PATH:{result_file}")
        return result_obj, str(result_file)

    finally:
        pass


def main():
    load_dotenv_if_present()
    parser = argparse.ArgumentParser(description="Run ASR tests using Bundle Contract v1")
    parser.add_argument(
        "--model", type=str, required=True, help="Model ID (registered in harness.registry)"
    )

    # Mutually exclusive: --dataset or --audio/--input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--dataset", type=str, help="Dataset ID (smoke, primary, llm_primary, conversation)"
    )
    input_group.add_argument("--audio", type=str, help="Single audio/video file path (adhoc mode)")
    input_group.add_argument(
        "--input", type=str, dest="audio", help="Alias for --audio (accepts audio or video)"
    )

    parser.add_argument(
        "--device", type=str, default=None, help="Override device (e.g., cpu, mps, cuda)"
    )
    parser.add_argument(
        "--pre",
        type=str,
        default=None,
        help="Preprocessing operators (comma-separated, e.g., trim_silence,normalize_loudness)",
    )

    args = parser.parse_args()

    try:
        if args.audio:
            # Adhoc mode: single file (audio or video)
            result, artifact_path = run_asr_adhoc(
                args.model, args.audio, device=args.device, pre=args.pre
            )
            print("\n🎉 Adhoc run completed!")
            print(f"Artifact: {artifact_path}")
        else:
            # Dataset mode: existing behavior
            run_asr_test(args.model, args.dataset, device=args.device)
            print("\n🎉 Test completed successfully!")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
