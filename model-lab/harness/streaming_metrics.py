"""
Streaming latency measurement utilities (LCS-Y).

Dependency-free metrics for asr_stream models.
Computes first_token_latency, partial_update_rate, finalize_latency, RTF.
"""

from __future__ import annotations

import time
from typing import Any
import numpy as np


def _resolve_stream_ops(asr_stream_ns: dict[str, Any]) -> dict[str, Any]:
    """Resolve compatibility aliases across stream namespaces."""
    start = asr_stream_ns.get("start") or asr_stream_ns.get("start_stream")
    push_audio = asr_stream_ns.get("push_audio")
    finalize = asr_stream_ns.get("finalize")
    close = asr_stream_ns.get("close")
    get_transcript = asr_stream_ns.get("get_transcript")

    if start is None or push_audio is None or finalize is None:
        raise KeyError("asr_stream namespace missing required start/push_audio/finalize method(s)")

    return {
        "start": start,
        "push_audio": push_audio,
        "get_transcript": get_transcript,
        "finalize": finalize,
        "close": close,
    }


def measure_streaming_latency(
    asr_stream_ns: dict[str, Any],
    audio: np.ndarray,
    sr: int,
    *,
    chunk_ms: int = 160,
    frame_ms: int = 20,
    warmup_s: float = 0,
) -> dict[str, Any]:
    """
    Measure latency metrics for a streaming ASR model.
    
    Args:
        asr_stream_ns: The asr_stream namespace from a loaded bundle.
                       Must have: start, push_audio, get_transcript, finalize
        audio: Audio waveform as numpy array (float32)
        sr: Sample rate in Hz
        chunk_ms: Chunk size in milliseconds for pushing audio
        frame_ms: Frame size in milliseconds (for rate calculations)
        warmup_s: Seconds to skip at start (for model warmup)
    
    Returns:
        dict with:
            - first_token_latency_ms: Time until first non-empty partial/final
            - partial_update_rate_hz: Partials per second (0 if <2 partials)
            - finalize_latency_ms: Time spent in finalize() call
            - real_time_factor: Total processing time / audio duration
            - num_events: Total events observed
            - num_partials: Number of partial updates
            - num_finals: Number of final events
            - audio_duration_s: Duration of input audio
    """
    # Validate inputs
    audio = np.asarray(audio, dtype=np.float32)
    audio_duration_s = len(audio) / sr
    
    # Calculate chunk sizes
    chunk_samples = int(sr * chunk_ms / 1000)
    warmup_samples = int(sr * warmup_s)
    
    # Skip warmup
    if warmup_samples > 0:
        audio = audio[warmup_samples:]
        audio_duration_s = len(audio) / sr
    
    # Split audio into chunks
    chunks = []
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        if len(chunk) > 0:
            chunks.append(chunk)
    
    # Metrics tracking
    events = []
    partials = []
    finals = []
    first_token_time = None
    t_first_partial = None
    t_last_partial = None
    
    ops = _resolve_stream_ops(asr_stream_ns)

    # Start streaming
    handle = ops["start"](sr=sr)
    
    # Record start time
    t0 = time.perf_counter()
    
    # Push chunks and collect events
    for i, chunk in enumerate(chunks):
        push_events = ops["push_audio"](handle, chunk)

        # Get current transcript when supported; otherwise infer from push events.
        result = None
        if ops["get_transcript"] is not None:
            result = ops["get_transcript"](handle)
        else:
            if isinstance(push_events, list) and push_events:
                last = push_events[-1]
                if isinstance(last, dict):
                    result = {
                        "text": last.get("text", ""),
                        "is_final": bool(last.get("is_final") or last.get("type") == "final"),
                    }

        t_now = time.perf_counter()
        
        if result:
            events.append({
                "time": t_now - t0,
                "result": result,
            })
            
            # Check for first token
            text = result.get("text", "")
            is_final = result.get("is_final", False)
            
            if text and first_token_time is None:
                first_token_time = t_now - t0
            
            if is_final:
                finals.append(t_now - t0)
            else:
                # Partial update
                if text:  # Only count non-empty partials
                    partials.append(t_now - t0)
                    if t_first_partial is None:
                        t_first_partial = t_now - t0
                    t_last_partial = t_now - t0
    
    # Measure finalize latency separately
    t_finalize_start = time.perf_counter()
    final_result = ops["finalize"](handle)
    t_finalize_end = time.perf_counter()
    finalize_latency_ms = (t_finalize_end - t_finalize_start) * 1000
    
    # Record final if it has content
    if final_result:
        t_final = t_finalize_end - t0
        events.append({
            "time": t_final,
            "result": final_result,
        })
        if final_result.get("is_final", True):
            finals.append(t_final)
        
        # Check for first token in final
        text = final_result.get("text", "")
        if text and first_token_time is None:
            first_token_time = t_final
    
    # Total processing time
    t_end = time.perf_counter()
    total_time_s = t_end - t0
    
    # Calculate metrics
    
    # First token latency
    first_token_latency_ms = first_token_time * 1000 if first_token_time else None
    
    # Partial update rate
    if len(partials) >= 2 and t_last_partial > t_first_partial:
        partial_update_rate_hz = (len(partials) - 1) / (t_last_partial - t_first_partial)
    else:
        partial_update_rate_hz = 0.0
    
    # Real-time factor
    real_time_factor = total_time_s / audio_duration_s if audio_duration_s > 0 else 0.0
    
    result_metrics = {
        # Primary metrics
        "first_token_latency_ms": first_token_latency_ms,
        "partial_update_rate_hz": round(partial_update_rate_hz, 2),
        "finalize_latency_ms": round(finalize_latency_ms, 3),
        "real_time_factor": round(real_time_factor, 4),
        # Debug fields
        "num_events": len(events),
        "num_partials": len(partials),
        "num_finals": len(finals),
        "audio_duration_s": round(audio_duration_s, 3),
    }
    if ops["close"] is not None:
        try:
            ops["close"](handle)
        except Exception:
            pass
    return result_metrics


def format_latency_report(metrics: dict[str, Any]) -> str:
    """Format metrics as a human-readable report."""
    lines = [
        "Streaming Latency Report",
        "=" * 40,
        f"First Token Latency: {metrics['first_token_latency_ms']:.1f} ms" if metrics['first_token_latency_ms'] else "First Token Latency: N/A",
        f"Partial Update Rate: {metrics['partial_update_rate_hz']:.2f} Hz",
        f"Finalize Latency:    {metrics['finalize_latency_ms']:.3f} ms",
        f"Real-Time Factor:    {metrics['real_time_factor']:.4f}x",
        "-" * 40,
        f"Audio Duration:      {metrics['audio_duration_s']:.3f} s",
        f"Events:              {metrics['num_events']}",
        f"Partials:            {metrics['num_partials']}",
        f"Finals:              {metrics['num_finals']}",
    ]
    return "\n".join(lines)
