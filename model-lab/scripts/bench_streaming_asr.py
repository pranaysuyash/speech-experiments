#!/usr/bin/env python3
"""
Streaming ASR benchmark.

Purpose:
- Measure chunked streaming behavior (latency/RTF) using harness.streaming_asr.stream_asr.
- Produce a JSON report that can be referenced from PERFORMANCE_RESULTS.md.

This intentionally does not depend on the FastAPI server; it exercises the
streaming provider pipeline directly in-process.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Optional

MODEL_LAB_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_LAB_DIR))


def _load_audio_mono_16k(path: Path, target_sr: int = 16000) -> tuple[list[float], int]:
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(path)
    if hasattr(audio, "ndim") and audio.ndim > 1:
        audio = audio.mean(axis=1)

    if int(sr) != int(target_sr):
        # Avoid torch/torchaudio dependency for this benchmark.
        import librosa

        audio = librosa.resample(audio.astype("float32"), orig_sr=int(sr), target_sr=int(target_sr))
        sr = int(target_sr)

    audio = np.asarray(audio, dtype="float32")
    return audio.tolist(), int(sr)


def _float_to_pcm16_bytes(audio: list[float]) -> bytes:
    import numpy as np

    a = np.asarray(audio, dtype="float32")
    a = np.clip(a, -1.0, 1.0)
    i16 = (a * 32767.0).astype(np.int16)
    return i16.tobytes()


@dataclass
class ChunkStat:
    chunk_index: int
    chunk_start_s: float
    wall_start_s: float
    wall_end_s: float
    events: int
    chars: int
    first_event_latency_ms: Optional[float]


class _ChunkStream:
    def __init__(self, pcm: bytes, sample_rate: int, chunk_seconds: int, realtime: bool):
        self._pcm = pcm
        self._sample_rate = sample_rate
        self._chunk_seconds = int(chunk_seconds)
        self._realtime = realtime

        self.chunk_bytes = int(self._sample_rate * self._chunk_seconds * 2)
        if self.chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be > 0")

        self.current_chunk_index: int = -1
        self.current_chunk_wall_start: float = 0.0

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[bytes]:
        total = len(self._pcm)
        i = 0
        chunk_index = 0
        while i < total:
            self.current_chunk_index = chunk_index
            self.current_chunk_wall_start = time.perf_counter()
            chunk = self._pcm[i : i + self.chunk_bytes]
            i += self.chunk_bytes
            yield chunk
            chunk_index += 1
            if self._realtime:
                await asyncio.sleep(self._chunk_seconds)


async def _run_bench(
    audio_path: Path,
    chunk_seconds: int,
    sample_rate: int,
    realtime: bool,
    source: Optional[str],
) -> dict[str, Any]:
    # Import here so a plain `python ...` shows a clean error if harness isn't on path.
    from harness.streaming_asr import stream_asr

    audio, sr = _load_audio_mono_16k(audio_path, target_sr=sample_rate)
    pcm = _float_to_pcm16_bytes(audio)

    stream = _ChunkStream(pcm, sample_rate=sr, chunk_seconds=chunk_seconds, realtime=realtime)
    total_audio_s = len(pcm) / (sr * 2)

    chunk_stats: dict[int, ChunkStat] = {}
    first_event_seen_for_chunk: set[int] = set()

    t0 = time.perf_counter()
    events_total = 0
    chars_total = 0

    async for event in stream_asr(stream, sample_rate=sr, source=source):
        now = time.perf_counter()
        events_total += 1
        text = str(event.get("text") or "")
        chars_total += len(text)

        idx = stream.current_chunk_index
        if idx not in chunk_stats:
            chunk_stats[idx] = ChunkStat(
                chunk_index=idx,
                chunk_start_s=idx * float(chunk_seconds),
                wall_start_s=stream.current_chunk_wall_start,
                wall_end_s=now,
                events=0,
                chars=0,
                first_event_latency_ms=None,
            )

        stat = chunk_stats[idx]
        stat.wall_end_s = now
        stat.events += 1
        stat.chars += len(text)

        if idx not in first_event_seen_for_chunk:
            first_event_seen_for_chunk.add(idx)
            stat.first_event_latency_ms = (now - stat.wall_start_s) * 1000.0

    t1 = time.perf_counter()

    # Derive per-chunk durations from wall_start deltas (only meaningful for non-realtime runs).
    ordered = [chunk_stats[i] for i in sorted(chunk_stats.keys())]
    chunk_wall_durations_ms: list[float] = []
    for i in range(len(ordered) - 1):
        chunk_wall_durations_ms.append((ordered[i + 1].wall_start_s - ordered[i].wall_start_s) * 1000.0)
    if ordered:
        chunk_wall_durations_ms.append((t1 - ordered[-1].wall_start_s) * 1000.0)

    wall_s = t1 - t0
    rtf = wall_s / max(1e-9, total_audio_s)

    return {
        "task": "streaming_asr_benchmark",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": {
            "path": str(audio_path),
            "sample_rate": sr,
            "chunk_seconds": int(chunk_seconds),
            "realtime": bool(realtime),
            "source": source,
            "duration_seconds": total_audio_s,
        },
        "env": {
            "MODEL_LAB_ASR_PROVIDER": os.getenv("MODEL_LAB_ASR_PROVIDER", ""),
            "MODEL_LAB_WHISPER_MODEL": os.getenv("MODEL_LAB_WHISPER_MODEL", ""),
            "MODEL_LAB_WHISPER_DEVICE": os.getenv("MODEL_LAB_WHISPER_DEVICE", ""),
            "MODEL_LAB_WHISPER_COMPUTE": os.getenv("MODEL_LAB_WHISPER_COMPUTE", ""),
            "MODEL_LAB_ASR_VAD": os.getenv("MODEL_LAB_ASR_VAD", ""),
        },
        "summary": {
            "wall_seconds": wall_s,
            "rtf": rtf,
            "events_total": events_total,
            "chars_total": chars_total,
            "chunks_with_events": len(chunk_stats),
            "chunk_wall_durations_ms": {
                "min": min(chunk_wall_durations_ms) if chunk_wall_durations_ms else None,
                "p50": sorted(chunk_wall_durations_ms)[len(chunk_wall_durations_ms) // 2]
                if chunk_wall_durations_ms
                else None,
                "max": max(chunk_wall_durations_ms) if chunk_wall_durations_ms else None,
            },
        },
        "chunks": [asdict(s) for s in ordered],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to input audio (wav/flac/etc)")
    p.add_argument("--out", help="Output JSON path (default: runs/streaming_bench/<timestamp>.json)")
    p.add_argument("--chunk-seconds", type=int, default=4)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--realtime", action="store_true", help="Sleep chunk_seconds between chunks")
    p.add_argument("--source", choices=["mic", "system"], help="Optional source tag")
    args = p.parse_args()

    audio_path = Path(args.input).expanduser().resolve()
    if not audio_path.exists():
        raise SystemExit(f"Input audio not found: {audio_path}")

    report = asyncio.run(
        _run_bench(
            audio_path=audio_path,
            chunk_seconds=args.chunk_seconds,
            sample_rate=args.sample_rate,
            realtime=args.realtime,
            source=args.source,
        )
    )

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        runs_dir = (MODEL_LAB_DIR / "runs" / "streaming_bench").resolve()
        runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = report["timestamp_utc"].replace(":", "").replace("-", "")
        out_path = runs_dir / f"streaming_asr_{stamp}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
