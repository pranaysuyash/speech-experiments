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
    
    Returns result with WER/CER and RTF.
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
    wall_s = time.perf_counter() - t_start
    
    hypothesis = result.get("text", "")
    rtf = wall_s / duration_s if duration_s > 0 else 0
    
    # Compute WER/CER if reference provided
    metrics = {
        "rtf": round(rtf, 4),
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
        timing={"wall_s": round(wall_s, 3), "rtf": round(rtf, 4)},
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


def format_result_table(results: list[dict[str, Any]]) -> str:
    """Format multiple results as a comparison table."""
    if not results:
        return "No results."
    
    lines = [
        "| Model | Surface | RTF | WER | CER | First Token |",
        "|-------|---------|-----|-----|-----|-------------|",
    ]
    
    for r in results:
        model = r.get("model_id", "?")
        surface = r.get("surface", "?")
        rtf = r.get("metrics", {}).get("rtf", r.get("timing", {}).get("rtf", "?"))
        wer = r.get("metrics", {}).get("wer", "-")
        cer = r.get("metrics", {}).get("cer", "-")
        first_token = r.get("metrics", {}).get("first_token_latency_ms", "-")
        
        if isinstance(rtf, float):
            rtf = f"{rtf:.3f}"
        if isinstance(wer, float):
            wer = f"{wer:.3f}"
        if isinstance(cer, float):
            cer = f"{cer:.3f}"
        if isinstance(first_token, float):
            first_token = f"{first_token:.1f}ms"
        
        lines.append(f"| {model} | {surface} | {rtf} | {wer} | {cer} | {first_token} |")
    
    return "\n".join(lines)
