#!/usr/bin/env python3
"""
Session Runner CLI - Orchestrate Audio NLP Pipeline.

Usage:
  python scripts/run_session.py --input meeting.mp4
  python scripts/run_session.py --input meeting.mp4 --resume-from runs/sessions/hash/runid
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path

# Add harness to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.session import SessionRunner
from harness.media_ingest import IngestConfig

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("run_session")


def _console_url(run_id: str, port: int) -> str:
    """Generate console URL for a run."""
    return f"http://localhost:{port}/runs/{run_id}"


def _emit_run_result(run_id: str, run_dir: str, console_port: int) -> None:
    """Emit machine-parseable result sentinel (one-line, compact JSON)."""
    payload = {
        "run_id": run_id,
        "run_dir": run_dir,
        "console_url": _console_url(run_id, console_port)
    }
    print("------------------------------------------------------------")
    # Compact JSON on one line for bomb-proof parsing
    print("RUN_SESSION_RESULT=" + json.dumps(payload, separators=(",", ":"), ensure_ascii=False), flush=True)
    print()
    print("=" * 60)
    print("✅ Meeting processed successfully!")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run deterministic audio session pipeline")
    p.add_argument("--input", required=True, help="Path to input media (audio or video)")
    p.add_argument("--out-dir", help="Base output directory (default: $MODEL_LAB_RUNS_ROOT or 'runs')")
    p.add_argument("--force", action="store_true", help="Force recompute steps")
    p.add_argument("--no-resume", action="store_true", help="Disable resume")
    p.add_argument("--steps", nargs="*", default=None, help="Subset of steps to run")
    p.add_argument("--normalize", action="store_true", help="Enable loudness normalization")
    p.add_argument("--trim-silence", action="store_true", help="Enable conservative silence trimming")
    p.add_argument("--pre", action="store_true", help="Enable standard preprocessing suite")
    p.add_argument("--resume-from", help="Resume from specific run directory")
    p.add_argument("--asr-model", default="faster_whisper", help="ASR model type")
    p.add_argument("--asr-size", default="large-v3", help="ASR model size/path")
    p.add_argument("--compute-type", default="float16", help="Compute type (float16, int8, float32)")
    p.add_argument("--diarization-model", default="pyannote_diarization", help="Diarization model to use")
    p.add_argument("--console-port", type=int, default=5174, help="Frontend console port (default: 5174)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Preprocessing Config
    normalize = args.normalize
    trim_silence = args.trim_silence
    if args.pre:
        normalize = True
        trim_silence = True

    ingest_cfg = IngestConfig(
        normalize=normalize,
        trim_silence=trim_silence,
        loudnorm_mode="single_pass",
    )
    
    # Extra config for legacy runners
    config = {
        "asr": {
            "model_type": args.asr_model,
            "model_name": args.asr_size,
            "inference": {
                "compute_type": args.compute_type
            }
        },
        "diarization": {
            "model": args.diarization_model,
        },
        "resume_from": args.resume_from
    }

    # Default output dir to standard runs root if not specified
    out_dir_str = args.out_dir or os.environ.get("MODEL_LAB_RUNS_ROOT", "runs")
    
    runner = SessionRunner(
        input_path=Path(args.input),
        output_dir=Path(out_dir_str),
        force=args.force,
        resume=not args.no_resume,
        preprocessing=ingest_cfg,
        steps=args.steps,
        config=config
    )
    runner.run()
    
    # Emit machine-parsable result for wrapper scripts
    _emit_run_result(
        run_id=runner.run_id,
        run_dir=str(runner.session_dir),
        console_port=args.console_port
    )


if __name__ == "__main__":
    main()
