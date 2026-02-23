#!/usr/bin/env python3
"""
Headless TTS runner for production testing.
Uses Bundle Contract v1 - no per-model special casing.

Usage: uv run python -m scripts.run_tts --model lfm2_5_audio --dataset tts_smoke_v1
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
import soundfile as sf

# Add parent to path for harness imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.env import load_dotenv_if_present

from harness.metrics_tts import detect_audio_issues
from harness.registry import ModelRegistry
from harness.run_provenance import create_provenance, create_run_context
from harness.timers import PerformanceTimer


def compute_text_sha256(text: str) -> str:
    """Compute SHA256 hash of text content."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def compute_file_sha256(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def load_model_config(model_id: str) -> dict:
    """Load model configuration."""
    config_path = Path(f"models/{model_id}/config.yaml")

    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
            config.setdefault("model_type", model_id)
            return config

    return {"model_type": model_id, "model_name": model_id, "audio": {"sample_rate": 24000}}


def load_tts_dataset(dataset: str) -> dict:
    """Load TTS dataset definition."""
    dataset_path = Path(f"data/golden/{dataset}.yaml")

    if not dataset_path.exists():
        raise ValueError(f"Dataset not found: {dataset_path}")

    with open(dataset_path) as f:
        return yaml.safe_load(f)


def run_tts_test(model_id: str, dataset: str, device: str = None):
    """
    Run TTS test using Bundle Contract v1.

    Tests all prompts in dataset, computes audio health gates.
    """
    print(f"=== TTS Test: {model_id} on {dataset} ===")

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

    # Verify TTS capability exists
    if "tts" not in bundle["capabilities"]:
        raise ValueError(f"Model {model_id} does not have TTS capability")

    # Get the synthesize function from bundle
    synthesize_fn = bundle["tts"]["synthesize"]

    # Load dataset
    dataset_def = load_tts_dataset(dataset)
    prompts = dataset_def["prompts"]
    print(f"Dataset: {dataset} ({len(prompts)} prompts)")

    # Create output directories
    runs_dir = Path(f"runs/{model_id}/tts")
    audio_dir = runs_dir / "audio"
    runs_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Process each prompt
    results = []
    total_latency = 0
    total_duration = 0
    all_healthy = True

    for prompt in prompts:
        prompt_id = prompt["id"]
        text = prompt["text"]
        language = prompt.get("language", "en")

        print(f"\nPrompt: {prompt_id}")
        print(f"  Text: {text[:50]}...")

        timer = PerformanceTimer()

        with timer.time_operation(f"{model_id}_synthesize_{prompt_id}") as timing_container:
            result = synthesize_fn(text, language=language)
            audio = result.get("audio")
            sr = result.get("sample_rate", 24000)

        timing = timing_container["result"]
        latency_ms = timing.elapsed_time_ms

        # Convert audio to numpy if needed
        if hasattr(audio, "cpu"):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = np.array(audio)

        # Ensure correct shape (flatten if needed)
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()

        duration_s = len(audio_np) / sr
        rtf = (latency_ms / 1000) / duration_s if duration_s > 0 else float("inf")

        # Compute audio health gates
        audio_health = detect_audio_issues(audio_np, sr)

        # Save audio file
        audio_filename = f"{prompt_id}_{datetime.now().strftime('%H%M%S')}.wav"
        audio_path = audio_dir / audio_filename

        # Save as WAV (avoid torchaudio/ffmpeg dylib conflicts on macOS)
        sf.write(str(audio_path), audio_np, sr)

        audio_sha256 = compute_file_sha256(audio_path)

        prompt_result = {
            "prompt_id": prompt_id,
            "text": text,
            "text_sha256": compute_text_sha256(text),
            "language": language,
            "audio_file": str(audio_path.relative_to(runs_dir.parent.parent)),
            "audio_sha256": audio_sha256,
            "duration_s": duration_s,
            "sr": sr,
            "latency_ms": latency_ms,
            "rtf": rtf,
            "audio_health": audio_health,
        }
        results.append(prompt_result)

        total_latency += latency_ms
        total_duration += duration_s

        if audio_health["has_failure"]:
            all_healthy = False
            print(
                f"  ⚠️ Audio health failed: {[k for k, v in audio_health.items() if k.startswith('is_') or k.startswith('has_') and v]}"
            )
        else:
            print(f"  ✓ {duration_s:.2f}s @ {sr}Hz in {latency_ms:.0f}ms (RTF={rtf:.2f}x)")

    # Aggregate metrics
    avg_rtf = (total_latency / 1000) / total_duration if total_duration > 0 else 0

    # Build run result
    run_result = {
        "provider_id": model_id,
        "capability": "tts",
        "manifest": {
            "timestamp": datetime.now().isoformat(),
            "git_hash": "unknown",  # TODO: add git hash
            "schema_version": 1,
        },
        "input": {"dataset_id": dataset, "prompt_count": len(prompts)},
        "output": {"prompts": results, "total_duration_s": total_duration, "avg_rtf": avg_rtf},
        "metrics": {
            "latency_ms_total": total_latency,
            "latency_ms_avg": total_latency / len(prompts) if prompts else 0,
            "rtf": avg_rtf,
            "duration_s": total_duration,
        },
        "audio_health": {
            "all_healthy": all_healthy,
            "healthy_count": sum(1 for r in results if not r["audio_health"]["has_failure"]),
            "failed_count": sum(1 for r in results if r["audio_health"]["has_failure"]),
        },
        "system": {
            "device": bundle["device"],
            "model": model_name,
            "inference_type": "local",
            "capabilities": bundle["capabilities"],
        },
        "evidence": {
            "grade": "smoke",
            "dataset_id": dataset,
            "valid": all_healthy,
            "invalid_reasons": ["tts_audio_health_failed"] if not all_healthy else [],
        },
        "timestamps": {
            "started_at": datetime.now().isoformat(),
            "finished_at": datetime.now().isoformat(),
        },
        "errors": [],
    }

    # Add provenance - TTS smoke has NO ground truth
    run_result["provenance"] = create_provenance(
        dataset_id=dataset,
        dataset_path=None,
        audio_path=None,
        ground_truth_path=None,  # No reference for TTS
        metrics_valid=True,
    )

    # Add run_context for interpretable metrics
    run_result["run_context"] = create_run_context(
        device=bundle["device"],
        audio_duration_s=total_duration,
    )

    # Save run JSON
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_file = runs_dir / f"{timestamp}.json"
    with open(result_file, "w") as f:
        json.dump(run_result, f, indent=2)

    # Summary file
    summary_file = runs_dir / "summary.json"
    summary = {
        "model_id": model_id,
        "dataset": dataset,
        "last_run": datetime.now().isoformat(),
        "all_healthy": all_healthy,
        "healthy_count": run_result["audio_health"]["healthy_count"],
        "failed_count": run_result["audio_health"]["failed_count"],
        "avg_rtf": avg_rtf,
        "total_duration_s": total_duration,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved to: {result_file}")
    print(f"✓ Summary updated: {summary_file}")

    # Final summary
    print("\n=== TTS Test Summary ===")
    print(f"Prompts: {len(prompts)}")
    print(f"Healthy: {run_result['audio_health']['healthy_count']}/{len(prompts)}")
    print(f"Avg RTF: {avg_rtf:.2f}x")
    print(f"Total Duration: {total_duration:.1f}s")

    return run_result


def main():
    load_dotenv_if_present()
    parser = argparse.ArgumentParser(description="Run TTS tests using Bundle Contract v1")
    parser.add_argument(
        "--model", type=str, required=True, help="Model ID (registered in harness.registry)"
    )
    parser.add_argument(
        "--dataset", type=str, default="tts_smoke_v1", help="Dataset ID (default: tts_smoke_v1)"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override device (e.g., cpu, mps, cuda)"
    )

    args = parser.parse_args()

    try:
        result = run_tts_test(args.model, args.dataset, device=args.device)
        rc = 0
        if result["audio_health"]["all_healthy"]:
            print("\n🎉 TTS test completed successfully!")
        else:
            print("\n⚠️ TTS test completed with audio health issues")
            rc = 1

        # Workaround: Kokoro native deps can crash during Python teardown on macOS.
        # Use hard-exit to preserve the computed run status for automation.
        if sys.platform == "darwin" and args.model == "kokoro_tts":
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(rc)

        return rc
    except Exception as e:
        print(f"\n❌ TTS test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
