#!/usr/bin/env python3
"""
Streaming ASR runner with dataset support.

Usage:
  uv run python scripts/run_asr_stream.py --model kyutai_streaming --dataset asr_smoke_v1
  uv run python scripts/run_asr_stream.py --model voxtral_realtime_2602 --audio file.wav
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_MAP = {
    "smoke": (
        Path("data/audio/SMOKE/conversation_2ppl_10s.wav"),
        None,
        "smoke",
    ),
    "asr_smoke_v1": (
        Path("data/audio/PRIMARY/llm_recording_pranay.wav"),
        Path("data/text/PRIMARY/llm.txt"),
        "smoke",
    ),
    "primary": (
        Path("data/audio/PRIMARY/llm_recording_pranay.wav"),
        Path("data/text/PRIMARY/llm.txt"),
        "golden_batch",
    ),
    "llm_primary": (
        Path("data/audio/PRIMARY/llm_recording_pranay.wav"),
        Path("data/text/PRIMARY/llm.txt"),
        "golden_batch",
    ),
    "ux_primary": (
        Path("data/audio/PRIMARY/ux_psychology_30s.wav"),
        Path("data/text/PRIMARY/ux_psychology_30s.txt"),
        "golden_batch",
    ),
}


def _detect_device(user_device: str | None) -> str:
    if user_device:
        return user_device

    try:
        import torch
    except Exception:
        return "cpu"

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_dataset(dataset_id: str) -> tuple[Path, str | None, str]:
    if dataset_id not in DATASET_MAP:
        supported = ", ".join(sorted(DATASET_MAP.keys()))
        raise ValueError(f"Unknown dataset: {dataset_id}. Available: {supported}")

    audio_path, text_path, grade = DATASET_MAP[dataset_id]
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    reference = None
    if text_path:
        if not text_path.exists():
            raise FileNotFoundError(f"Reference text not found: {text_path}")
        reference = text_path.read_text(encoding="utf-8").strip()

    return audio_path, reference, grade


def run_streaming_dataset(
    model_id: str,
    dataset_id: str,
    device: str,
    chunk_ms: int,
) -> tuple[dict, Path]:
    from bench.runner import run_streaming_asr_bench

    audio_path, reference, grade = resolve_dataset(dataset_id)

    result = run_streaming_asr_bench(
        model_id=model_id,
        audio_path=str(audio_path),
        reference=reference,
        chunk_ms=chunk_ms,
        device=device,
    )

    result["capability"] = "asr_stream"
    result["meta"] = {
        "task": "asr_stream",
        "model_id": model_id,
        "dataset_id": dataset_id,
        "generated_at": datetime.now().isoformat(),
    }
    result["evidence"] = {
        "grade": grade,
        "dataset_id": dataset_id,
    }
    result["provenance"] = {
        "has_ground_truth": reference is not None,
        "metrics_valid": True,
    }

    out_dir = Path(f"runs/{model_id}/asr_stream")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_id}_{result['run_id']}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result, out_path


def run_streaming_adhoc(
    model_id: str,
    audio_path: Path,
    reference: str | None,
    device: str,
    chunk_ms: int,
) -> tuple[dict, Path]:
    from bench.runner import run_streaming_asr_bench

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    result = run_streaming_asr_bench(
        model_id=model_id,
        audio_path=str(audio_path),
        reference=reference,
        chunk_ms=chunk_ms,
        device=device,
    )

    result["capability"] = "asr_stream"
    result["meta"] = {
        "task": "asr_stream",
        "model_id": model_id,
        "dataset_id": f"adhoc_{int(time.time())}",
        "generated_at": datetime.now().isoformat(),
    }
    result["evidence"] = {"grade": "adhoc"}
    result["provenance"] = {
        "has_ground_truth": reference is not None,
        "metrics_valid": True,
    }

    out_dir = Path(f"runs/{model_id}/asr_stream")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"adhoc_{result['run_id']}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result, out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run streaming ASR benchmark")
    parser.add_argument("--model", required=True, help="Model ID")
    parser.add_argument("--device", default=None, help="cpu | mps | cuda (auto if omitted)")
    parser.add_argument("--chunk-ms", type=int, default=160, help="Chunk size in milliseconds")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--dataset", help="Dataset ID")
    input_group.add_argument("--audio", type=Path, help="Audio file path (adhoc mode)")

    parser.add_argument(
        "--reference",
        help="Reference transcript text for adhoc mode (optional)",
    )

    args = parser.parse_args()
    device = _detect_device(args.device)

    try:
        if args.dataset:
            _, artifact_path = run_streaming_dataset(
                model_id=args.model,
                dataset_id=args.dataset,
                device=device,
                chunk_ms=args.chunk_ms,
            )
        else:
            _, artifact_path = run_streaming_adhoc(
                model_id=args.model,
                audio_path=args.audio,
                reference=args.reference,
                device=device,
                chunk_ms=args.chunk_ms,
            )
    except Exception as exc:
        print(f"❌ Streaming ASR run failed: {exc}")
        return 1

    print("✓ Streaming ASR run completed")
    print(f"ARTIFACT_PATH:{artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
