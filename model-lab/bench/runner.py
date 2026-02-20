"""
Benchmark runner (LCS-B1+).

Run benchmarks for any surface, output JSON results.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Fix MPS threading on macOS: fork copies mutexes in invalid state
# Must use 'spawn' start method for MPS compatibility
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set


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
    status: str = "ok",
) -> dict[str, Any]:
    """
    Create standardized benchmark result.

    Schema:
    {
      "run_id": "...",
      "timestamp": "...",
      "model_id": "...",
      "surface": "...",
      "status": "ok",
      "input": {"path": "...", "duration_s": 5.0, "sr": 16000},
      "metrics": {...},
      "timing": {"rtf": 0.9, "wall_s": 4.5},
      "env": {"device": "mps", "runtime": "pytorch"}
    }
    """
    env["venv"] = os.environ.get("VIRTUAL_ENV", sys.prefix)
    return {
        "run_id": get_run_id(),
        "timestamp": datetime.now().isoformat(),
        "model_id": model_id,
        "surface": surface,
        "status": status,
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

    # Resolve compatible stream surface names
    stream_ns = bundle["asr_stream"]
    start = stream_ns.get("start") or stream_ns.get("start_stream")
    push_audio = stream_ns.get("push_audio")
    finalize = stream_ns.get("finalize")
    close = stream_ns.get("close")
    if start is None or push_audio is None or finalize is None:
        raise ValueError(f"{model_id} asr_stream missing required start/push_audio/finalize methods")

    # Run latency measurement
    t_start = time.perf_counter()
    latency_metrics = measure_streaming_latency(stream_ns, audio, sr, chunk_ms=chunk_ms)
    wall_s = time.perf_counter() - t_start

    # Get final transcript from a fresh run for WER
    handle = start(sr=sr)

    chunk_samples = int(sr * chunk_ms / 1000)
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i : i + chunk_samples]
        if len(chunk) > 0:
            push_audio(handle, chunk)

    final = finalize(handle)
    if close is not None:
        try:
            close(handle)
        except Exception:
            pass
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
        metrics["wer"] = wer_result[0] if isinstance(wer_result, tuple) else wer_result.get("wer")
        cer_result = asr_metrics.calculate_cer(reference, hypothesis)
        metrics["cer"] = float(cer_result)
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
    chunk_length_s: int | None = None,
    stride_length_s: tuple[int, int] | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """
    Run batch ASR benchmark.

    Returns result with WER/CER, RTF, and additional metrics.
    """
    import soundfile as sf
    import time

    from harness.registry import ModelRegistry
    from harness.metrics_asr import ASRMetrics

    # Load audio
    audio, sr = sf.read(audio_path)
    audio = np.asarray(audio, dtype=np.float32)
    duration_s = len(audio) / sr

    # Prepare config
    config = {}
    if chunk_length_s:
        config.setdefault("inference", {})["chunk_length_s"] = chunk_length_s
    if stride_length_s:
        config.setdefault("inference", {})["stride_length_s"] = stride_length_s
    if batch_size:
        config.setdefault("inference", {})["batch_size"] = batch_size

    # Load model
    bundle = ModelRegistry.load_model(model_id, config, device=device)

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
        metrics["wer"] = wer_result[0] if isinstance(wer_result, tuple) else wer_result.get("wer")
        cer_result = asr_metrics.calculate_cer(reference, hypothesis)
        metrics["cer"] = float(cer_result)

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


def load_results(
    results_dir: str = "bench/results", surface: str | None = None
) -> list[dict[str, Any]]:
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


def _run_single_asr_bench(
    model_id: str, audio_path: str, reference: str | None, device: str
) -> dict[str, Any]:
    """Helper to run benchmark in a separate process."""
    return run_batch_asr_bench(model_id, audio_path, reference=reference, device=device)


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
        model_list = list_models_by_filter(
            surface="asr", device=device, ci=ci_only if ci_only else None
        )
        models = [m["model_id"] for m in model_list]

    results = []

    # Use spawn context to ensure clean MPS state for each model
    # fork() is unsafe with MPS/Torch
    ctx = multiprocessing.get_context("spawn")

    with ctx.Pool(1, maxtasksperchild=1) as pool:
        for model_id in models:
            try:
                # Run in separate process to isolate MPS state
                # Must use top-level function for pickling
                result = pool.apply(
                    _run_single_asr_bench, args=(model_id, audio_path, reference, device)
                )
                save_result(result)
                results.append(result)
            except Exception as e:
                # Record failure
                results.append(
                    {
                        "model_id": model_id,
                        "surface": "asr",
                        "error": str(e),
                    }
                )

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


def run_enhance_bench(
    model_id: str,
    noisy_path: str,
    clean_path: str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Run enhancement benchmark.

    Args:
        model_id: Model ID (rnnoise, deepfilternet, etc.)
        noisy_path: Path to noisy audio
        clean_path: Optional path to clean reference for SI-SNR delta
        device: Device to use

    Returns:
        Benchmark result with enhancement metrics
    """
    import soundfile as sf
    import numpy as np
    import time

    from harness.registry import ModelRegistry
    from harness.metrics_enhance import si_snr, stoi, is_stoi_available

    # Load noisy audio
    noisy, sr = sf.read(noisy_path)
    noisy = np.asarray(noisy, dtype=np.float32)
    duration_s = len(noisy) / sr

    # Load clean reference if provided
    clean = None
    if clean_path is not None:
        clean, _ = sf.read(clean_path)
        clean = np.asarray(clean, dtype=np.float32)

    # Load model
    bundle = ModelRegistry.load_model(model_id, {}, device=device)

    if "enhance" not in bundle.get("capabilities", []):
        raise ValueError(f"{model_id} does not support enhance")

    # Run enhancement
    t_start = time.perf_counter()
    result = bundle["enhance"]["enhance"](noisy, sr=sr)
    t_end = time.perf_counter()
    wall_s = t_end - t_start
    wall_ms = wall_s * 1000
    rtf = wall_s / duration_s if duration_s > 0 else 0

    # Extract audio from result dict
    enhanced = result["audio"] if isinstance(result, dict) else result

    # Compute metrics
    metrics = {
        "rtf": round(rtf, 4),
        "wall_ms": round(wall_ms, 1),
        "audio_duration_s": round(duration_s, 3),
    }

    # Sanity checks
    input_energy = float(np.mean(noisy**2))
    output_energy = float(np.mean(enhanced**2))
    metrics["energy_ratio"] = round(output_energy / (input_energy + 1e-10), 4)
    metrics["output_silent"] = output_energy < 1e-10
    metrics["clipping_detected"] = float(np.max(np.abs(enhanced))) > 1.0

    if clean is not None:
        # Compute SI-SNR before and after
        min_len = min(len(clean), len(noisy), len(enhanced))
        si_snr_input = si_snr(clean[:min_len], noisy[:min_len])
        si_snr_output = si_snr(clean[:min_len], enhanced[:min_len])
        metrics["si_snr_input_db"] = round(si_snr_input, 2)
        metrics["si_snr_output_db"] = round(si_snr_output, 2)
        metrics["si_snr_delta_db"] = round(si_snr_output - si_snr_input, 2)

        # Optional STOI
        if is_stoi_available():
            stoi_result = stoi(clean[:min_len], enhanced[:min_len], sr=sr)
            if not stoi_result.get("skipped"):
                metrics["stoi"] = round(stoi_result["score"], 4)

    return create_result_schema(
        model_id=model_id,
        surface="enhance",
        input_info={
            "path": noisy_path,
            "duration_s": round(duration_s, 3),
            "sr": sr,
            "has_clean_ref": clean is not None,
        },
        metrics=metrics,
        timing={"wall_s": round(wall_s, 3), "wall_ms": round(wall_ms, 1), "rtf": round(rtf, 4)},
        env={"device": device, "model_type": bundle.get("model_type", "")},
    )


def run_classify_bench(
    model_id: str,
    audio_path: str,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run classification benchmark."""
    import soundfile as sf
    from harness.registry import ModelRegistry

    audio, sr = sf.read(audio_path)
    audio = np.asarray(audio, dtype=np.float32)
    duration_s = len(audio) / sr

    bundle = ModelRegistry.load_model(model_id, {}, device=device)
    if "classify" not in bundle.get("capabilities", []):
        raise ValueError(f"{model_id} does not support classify")

    t_start = time.perf_counter()
    result = bundle["classify"]["classify"](audio, sr=sr)
    t_end = time.perf_counter()
    wall_s = t_end - t_start

    labels = result.get("labels", [])[:5]
    scores = result.get("scores", [])[:5]

    return create_result_schema(
        model_id=model_id,
        surface="classify",
        input_info={"path": audio_path, "duration_s": round(duration_s, 3), "sr": sr},
        metrics={"top_5_labels": labels, "top_5_scores": [round(float(s), 4) for s in scores]},
        timing={"wall_s": round(wall_s, 3), "wall_ms": round(wall_s * 1000, 1)},
        env={"device": device},
    )


def run_embed_bench(model_id: str, audio_path: str, device: str = "cpu") -> dict[str, Any]:
    """Run embedding benchmark."""
    import soundfile as sf
    from harness.registry import ModelRegistry

    audio, sr = sf.read(audio_path)
    audio = np.asarray(audio, dtype=np.float32)
    duration_s = len(audio) / sr

    bundle = ModelRegistry.load_model(model_id, {}, device=device)
    if "embed" not in bundle.get("capabilities", []):
        raise ValueError(f"{model_id} does not support embed")

    t_start = time.perf_counter()
    embedding = bundle["embed"]["embed"](audio, sr=sr)
    t_end = time.perf_counter()
    wall_s = t_end - t_start

    return create_result_schema(
        model_id=model_id,
        surface="embed",
        input_info={"path": audio_path, "duration_s": round(duration_s, 3), "sr": sr},
        metrics={
            "embedding_dim": embedding.shape[-1] if hasattr(embedding, "shape") else len(embedding)
        },
        timing={"wall_s": round(wall_s, 3), "wall_ms": round(wall_s * 1000, 1)},
        env={"device": device},
    )


def run_separate_bench(model_id: str, audio_path: str, device: str = "cpu") -> dict[str, Any]:
    """Run source separation benchmark."""
    import soundfile as sf
    from harness.registry import ModelRegistry

    audio, sr = sf.read(audio_path)
    audio = np.asarray(audio, dtype=np.float32)
    duration_s = len(audio) / sr

    bundle = ModelRegistry.load_model(model_id, {}, device=device)
    if "separate" not in bundle.get("capabilities", []):
        raise ValueError(f"{model_id} does not support separate")

    t_start = time.perf_counter()
    result = bundle["separate"]["separate"](audio, sr=sr)
    t_end = time.perf_counter()
    wall_s = t_end - t_start
    rtf = wall_s / duration_s if duration_s > 0 else 0

    stems_dict = result.get("stems", {})
    stems = list(stems_dict.keys())

    return create_result_schema(
        model_id=model_id,
        surface="separate",
        input_info={"path": audio_path, "duration_s": round(duration_s, 3), "sr": sr},
        metrics={"num_stems": len(stems), "stem_names": stems, "rtf": round(rtf, 4)},
        timing={
            "wall_s": round(wall_s, 3),
            "wall_ms": round(wall_s * 1000, 1),
            "rtf": round(rtf, 4),
        },
        env={"device": device},
    )


def run_tts_bench(model_id: str, text: str, device: str = "cpu") -> dict[str, Any]:
    """Run TTS benchmark with honest reporting (no placeholders)."""
    from harness.registry import ModelRegistry
    import numpy as np

    try:
        bundle = ModelRegistry.load_model(model_id, {}, device=device)
    except Exception as e:
        # If loader fails (e.g. auth/missing file), we skip/error gracefully
        print(f"Skipping {model_id}: {e}")
        return create_result_schema(
            model_id=model_id,
            surface="tts",
            input_info={"text": text[:50], "text_len": len(text)},
            metrics={"status": "skipped", "error": str(e)},
            timing={"wall_s": 0.0, "wall_ms": 0.0},
            env={"device": device},
        )

    if "tts" not in bundle.get("capabilities", []):
        raise ValueError(f"{model_id} does not support tts")

    t_start = time.perf_counter()
    try:
        result = bundle["tts"]["synthesize"](text)
    except Exception as e:
        t_end = time.perf_counter()
        return create_result_schema(
            model_id=model_id,
            surface="tts",
            input_info={"text": text[:50], "text_len": len(text)},
            metrics={"status": "error", "error": str(e)},
            timing={"wall_s": t_end - t_start, "wall_ms": (t_end - t_start) * 1000},
            env={"device": device},
        )
    t_end = time.perf_counter()
    wall_s = t_end - t_start

    # Check if the synthesizer returned an explicit status (e.g. blocked/skipped/error)
    if "status" in result and result["status"] != "ok":
        return create_result_schema(
            model_id=model_id,
            surface="tts",
            status=result["status"],
            input_info={"text": text[:50], "text_len": len(text)},
            metrics={"error": result.get("error", "Unknown error")},
            timing={"wall_s": wall_s, "wall_ms": wall_s * 1000},
            env={"device": device},
        )

    audio = result.get("audio", np.array([]))
    sr = result.get("sample_rate", 24000)

    if len(audio) == 0:
        return create_result_schema(
            model_id=model_id,
            surface="tts",
            status="error",
            input_info={"text": text[:50], "text_len": len(text)},
            metrics={"status": "error", "error": "Model produced empty audio"},
            timing={"wall_s": wall_s, "wall_ms": wall_s * 1000},
            env={"device": device},
        )

    duration_s = len(audio) / sr if sr > 0 else 0
    rtf = wall_s / duration_s if duration_s > 0 else 0

    # Compute audio quality metrics
    peak = float(np.max(np.abs(audio)))
    silence_thresh = 0.01
    silence_ratio = float(np.mean(np.abs(audio) < silence_thresh))

    return create_result_schema(
        model_id=model_id,
        surface="tts",
        status="ok",
        input_info={"text": text[:50], "text_len": len(text)},
        metrics={
            "audio_duration_s": round(duration_s, 3),
            "rtf": round(rtf, 4),
            "sr": sr,
            "silence_ratio": round(silence_ratio, 4),
            "peak": round(peak, 4),
        },
        timing={"wall_s": round(wall_s, 3), "wall_ms": round(wall_s * 1000, 1)},
        env={"device": device},
    )
