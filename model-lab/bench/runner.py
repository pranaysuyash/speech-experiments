"""
Benchmark runner (LCS-B1+).

Run benchmarks for any surface, output JSON results.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def get_run_id() -> str:
    """Generate unique run ID."""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def create_result_schema(
    model_id: str,
    surface: str,
    input_info: dict[str, Any],
    metrics: dict[str, Any],
    timing: dict[str, float],
    env: dict[str, str],
) -> dict[str, Any]:
    """
    Create standardized benchmark result.
    
    Schema:
    {
      "run_id": "...",
      "timestamp": "...",
      "model_id": "...",
      "surface": "...",
      "input": {"path": "...", "duration_s": 5.0, "sr": 16000},
      "metrics": {...},
      "timing": {"rtf": 0.9, "wall_s": 4.5},
      "env": {"device": "mps", "runtime": "pytorch"}
    }
    """
    return {
        "run_id": get_run_id(),
        "timestamp": datetime.now().isoformat(),
        "model_id": model_id,
        "surface": surface,
        "input": input_info,
        "metrics": metrics,
        "timing": timing,
        "env": env,
    }


def run_streaming_asr_bench(
    model_id: str,
    audio_path: str,
    reference: str | None = None,
    chunk_ms: int = 160,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Run streaming ASR benchmark.
    
    Returns result with latency metrics + WER/CER if reference provided.
    """
    import soundfile as sf
    
    from harness.registry import ModelRegistry
    from harness.streaming_metrics import measure_streaming_latency
    from harness.metrics_asr import ASRMetrics
    
    # Load audio
    audio, sr = sf.read(audio_path)
    audio = np.asarray(audio, dtype=np.float32)
    duration_s = len(audio) / sr
    
    # Load model
    bundle = ModelRegistry.load_model(model_id, {}, device=device)
    
    if "asr_stream" not in bundle.get("capabilities", []):
        raise ValueError(f"{model_id} does not support asr_stream")
    
    # Run latency measurement
    t_start = time.perf_counter()
    latency_metrics = measure_streaming_latency(
        bundle["asr_stream"], audio, sr, chunk_ms=chunk_ms
    )
    wall_s = time.perf_counter() - t_start
    
    # Get final transcript from a fresh run for WER
    handle = bundle["asr_stream"]["start"](sr=sr)
    
    chunk_samples = int(sr * chunk_ms / 1000)
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        if len(chunk) > 0:
            bundle["asr_stream"]["push_audio"](handle, chunk)
    
    final = bundle["asr_stream"]["finalize"](handle)
    hypothesis = final.get("text", "")
    
    # Compute WER/CER if reference provided
    metrics = {
        "first_token_latency_ms": latency_metrics["first_token_latency_ms"],
        "partial_update_rate_hz": latency_metrics["partial_update_rate_hz"],
        "finalize_latency_ms": latency_metrics["finalize_latency_ms"],
        "rtf": latency_metrics["real_time_factor"],
        "num_partials": latency_metrics["num_partials"],
    }
    
    if reference:
        asr_metrics = ASRMetrics()
        wer_result = asr_metrics.calculate_wer(reference, hypothesis)
        cer_result = asr_metrics.calculate_cer(reference, hypothesis)
        metrics["wer"] = wer_result["wer"]
        metrics["cer"] = cer_result["cer"]
        metrics["transcript"] = hypothesis
    
    return create_result_schema(
        model_id=model_id,
        surface="asr_stream",
        input_info={"path": audio_path, "duration_s": round(duration_s, 3), "sr": sr},
        metrics=metrics,
        timing={"wall_s": round(wall_s, 3), "rtf": latency_metrics["real_time_factor"]},
        env={"device": device, "model_type": bundle.get("model_type", "")},
    )


def run_batch_asr_bench(
    model_id: str,
    audio_path: str,
    reference: str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Run batch ASR benchmark.
    
    Returns result with WER/CER, RTF, and additional metrics.
    """
    import soundfile as sf
    
    from harness.registry import ModelRegistry
    from harness.metrics_asr import ASRMetrics
    
    # Load audio
    audio, sr = sf.read(audio_path)
    audio = np.asarray(audio, dtype=np.float32)
    duration_s = len(audio) / sr
    
    # Load model
    bundle = ModelRegistry.load_model(model_id, {}, device=device)
    
    if "asr" not in bundle.get("capabilities", []):
        raise ValueError(f"{model_id} does not support asr")
    
    # Run transcription
    t_start = time.perf_counter()
    result = bundle["asr"]["transcribe"](audio, sr=sr)
    t_end = time.perf_counter()
    wall_s = t_end - t_start
    wall_ms = wall_s * 1000
    
    hypothesis = result.get("text", "")
    rtf = wall_s / duration_s if duration_s > 0 else 0
    
    # Extract segments if available
    segments = result.get("segments", [])
    num_segments = len(segments) if segments else 1
    
    # Compute metrics
    metrics = {
        "rtf": round(rtf, 4),
        "wall_ms": round(wall_ms, 1),
        "audio_duration_s": round(duration_s, 3),
        "text_len": len(hypothesis),
        "num_segments": num_segments,
        "transcript": hypothesis,
    }
    
    if reference:
        asr_metrics = ASRMetrics()
        wer_result = asr_metrics.calculate_wer(reference, hypothesis)
        cer_result = asr_metrics.calculate_cer(reference, hypothesis)
        metrics["wer"] = wer_result["wer"]
        metrics["cer"] = cer_result["cer"]
    
    return create_result_schema(
        model_id=model_id,
        surface="asr",
        input_info={"path": audio_path, "duration_s": round(duration_s, 3), "sr": sr},
        metrics=metrics,
        timing={"wall_s": round(wall_s, 3), "wall_ms": round(wall_ms, 1), "rtf": round(rtf, 4)},
        env={"device": device, "model_type": bundle.get("model_type", "")},
    )


def save_result(result: dict[str, Any], output_dir: str = "bench/results") -> Path:
    """Save result to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"{result['model_id']}_{result['surface']}_{result['run_id']}.json"
    filepath = output_path / filename
    
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
    
    return filepath


def load_results(results_dir: str = "bench/results", surface: str | None = None) -> list[dict[str, Any]]:
    """Load all results from directory, optionally filtered by surface."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return []
    
    results = []
    for filepath in results_path.glob("*.json"):
        with open(filepath) as f:
            result = json.load(f)
            if surface is None or result.get("surface") == surface:
                results.append(result)
    
    return results


def run_batch_asr_sweep(
    audio_path: str,
    reference: str | None = None,
    device: str = "cpu",
    models: list[str] | None = None,
    ci_only: bool = True,
) -> list[dict[str, Any]]:
    """
    Run batch ASR benchmark across multiple models.
    
    Args:
        audio_path: Path to audio file
        reference: Optional reference text for WER
        device: Device to use (cpu, cuda, mps)
        models: List of model IDs, or None to use selector
        ci_only: If True and models is None, only include CI-safe models
    
    Returns:
        List of benchmark results
    """
    from harness.selector import list_models_by_filter
    
    if models is None:
        # Use selector to find ASR models
        model_list = list_models_by_filter(surface="asr", device=device, ci=ci_only if ci_only else None)
        models = [m["model_id"] for m in model_list]
    
    results = []
    for model_id in models:
        try:
            result = run_batch_asr_bench(model_id, audio_path, reference=reference, device=device)
            save_result(result)
            results.append(result)
        except Exception as e:
            # Record failure
            results.append({
                "model_id": model_id,
                "surface": "asr",
                "error": str(e),
            })
    
    return results


def format_result_table(results: list[dict[str, Any]], sort_by: str = "wer") -> str:
    """Format multiple results as a comparison table, sorted."""
    if not results:
        return "No results."
    
    # Filter out errors
    valid_results = [r for r in results if "error" not in r]
    
    # Sort by metric
    def get_sort_key(r):
        metrics = r.get("metrics", {})
        if sort_by == "wer":
            wer = metrics.get("wer", 999)
            rtf = metrics.get("rtf", 999)
            return (wer if isinstance(wer, (int, float)) else 999, rtf)
        elif sort_by == "rtf":
            return metrics.get("rtf", 999)
        return 0
    
    valid_results = sorted(valid_results, key=get_sort_key)
    
    lines = [
        "| Model | Surface | RTF | WER | CER | Wall (ms) | First Token |",
        "|-------|---------|-----|-----|-----|-----------|-------------|",
    ]
    
    for r in valid_results:
        model = r.get("model_id", "?")
        surface = r.get("surface", "?")
        metrics = r.get("metrics", {})
        timing = r.get("timing", {})
        
        rtf = metrics.get("rtf", timing.get("rtf", "-"))
        wer = metrics.get("wer", "-")
        cer = metrics.get("cer", "-")
        wall_ms = timing.get("wall_ms", "-")
        first_token = metrics.get("first_token_latency_ms", "-")
        
        if isinstance(rtf, float):
            rtf = f"{rtf:.3f}"
        if isinstance(wer, float):
            wer = f"{wer:.3f}"
        if isinstance(cer, float):
            cer = f"{cer:.3f}"
        if isinstance(wall_ms, float):
            wall_ms = f"{wall_ms:.0f}"
        if isinstance(first_token, float):
            first_token = f"{first_token:.1f}ms"
        
        lines.append(f"| {model} | {surface} | {rtf} | {wer} | {cer} | {wall_ms} | {first_token} |")
    
    # Add error entries at the end
    error_results = [r for r in results if "error" in r]
    for r in error_results:
        model = r.get("model_id", "?")
        error = r.get("error", "unknown")[:30]
        lines.append(f"| {model} | ERROR | - | - | - | - | {error}... |")
    
    return "\n".join(lines)


def generate_bench_report(surface: str = "asr", results_dir: str = "bench/results") -> str:
    """Generate a benchmark report for a surface."""
    results = load_results(results_dir, surface=surface)
    
    if not results:
        return f"No {surface} results found in {results_dir}/"
    
    header = f"# {surface.upper()} Benchmark Report\n\n"
    header += f"**{len(results)} runs**\n\n"
    
    table = format_result_table(results, sort_by="wer")
    
    return header + table

