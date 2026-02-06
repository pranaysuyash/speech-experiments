#!/usr/bin/env python3
"""
Recommend and Run - Thin client of the decision engine.

Usage:
    python scripts/recommend_and_run.py --audio my_file.wav --task asr
    python scripts/recommend_and_run.py --audio meeting.wav --task diarization
    python scripts/recommend_and_run.py --use-case meeting_analysis --audio meeting.wav

Selection Priority (NOT by evidence grade):
    1. RECOMMENDED models from decisions.json
    2. ACCEPTABLE models (with warning)
    3. BEST_EFFORT fallback (explicit warning, not written to evidence)

Flags:
    --policy strict|best_effort  (default: strict - only RECOMMENDED/ACCEPTABLE)
    --include-in-arsenal         (default: false - runs saved but excluded from arsenal)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_decisions() -> dict:
    """Load machine-readable decisions.json."""
    path = Path("docs/decisions.json")
    if not path.exists():
        print("⚠️  decisions.json not found. Run: python scripts/generate_decisions.py")
        return None
    with open(path) as f:
        return json.load(f)


def get_best_model_for_task(task: str, policy: str = "strict") -> tuple:
    """
    Get best model for a task based on DECISION OUTCOME (not evidence grade).

    Selection priority: RECOMMENDED > ACCEPTABLE > BEST_EFFORT

    Returns (model_id, outcome, reason, is_best_effort)
    """
    decisions = load_decisions()
    if not decisions:
        return None, None, "decisions.json not found", False

    tasks = decisions.get("tasks", {})
    if task not in tasks:
        return None, None, f"No decisions available for task: {task}", False

    task_data = tasks[task]
    models = task_data.get("models", [])

    # Find RECOMMENDED models
    recommended = [m for m in models if m["outcome"] == "recommended"]
    if recommended:
        best = recommended[0]  # Already sorted by score
        return (
            best["model_id"],
            "RECOMMENDED",
            f"Decision engine: RECOMMENDED ({best['evidence_grade']})",
            False,
        )

    # Find ACCEPTABLE models
    acceptable = [m for m in models if m["outcome"] == "acceptable"]
    if acceptable:
        best = acceptable[0]
        return (
            best["model_id"],
            "ACCEPTABLE",
            f"Decision engine: ACCEPTABLE ({best['evidence_grade']}) - with trade-offs",
            False,
        )

    # Nothing meets criteria
    if policy == "strict":
        return None, None, "No RECOMMENDED or ACCEPTABLE models for this task", False

    # BEST_EFFORT fallback
    best_by_grade = task_data.get("best_by_grade")
    if best_by_grade:
        model = next((m for m in models if m["model_id"] == best_by_grade), None)
        grade = model["evidence_grade"] if model else "unknown"
        return (
            best_by_grade,
            "BEST_EFFORT",
            f"⚠️ BEST_EFFORT: No models meet criteria. Using {best_by_grade} ({grade})",
            True,
        )

    return None, None, "No models available for this task", False


def get_pipeline_for_use_case(use_case: str, policy: str = "strict") -> tuple:
    """
    Get recommended pipeline for a use case.

    Returns (pipeline_dict, outcome, reason, is_best_effort)
    """
    decisions = load_decisions()
    if not decisions:
        return None, None, "decisions.json not found", False

    use_cases = decisions.get("use_cases", {})
    if use_case not in use_cases:
        available = list(use_cases.keys())
        return None, None, f"Unknown use case: {use_case}. Available: {available}", False

    uc = use_cases[use_case]

    if uc.get("evaluation_mode") == "pipeline":
        outcome = uc.get("outcome", "rejected")
        pipeline = uc.get("pipeline", {})

        if outcome == "recommended":
            return pipeline, "RECOMMENDED", "Pipeline fully available", False
        elif outcome == "acceptable":
            warns = uc.get("warn_reasons", [])
            return pipeline, "ACCEPTABLE", f"Pipeline partial: {'; '.join(warns)}", False
        else:
            if policy == "best_effort":
                # Return whatever partial pipeline exists
                return (
                    pipeline,
                    "BEST_EFFORT",
                    f"⚠️ Pipeline incomplete: {'; '.join(uc.get('fatal_reasons', []))}",
                    True,
                )
            return (
                None,
                "REJECTED",
                f"Pipeline not viable: {'; '.join(uc.get('fatal_reasons', []))}",
                False,
            )
    else:
        # Single model use case
        best = uc.get("best_model")
        if best:
            # Find the task from the use case config
            # For now, return as single-task pipeline
            return {"primary": best}, "RECOMMENDED", f"Single-model: {best}", False
        return None, None, "No viable model for use case", False


def run_task(task: str, model_id: str, audio_path: Path, **kwargs):
    """Run a task with the specified model and return results."""

    print(f"\n{'=' * 60}")
    print(f"  Task: {task.upper()}")
    print(f"  Model: {model_id}")
    print(f"  Audio: {audio_path}")
    print(f"{'=' * 60}\n")

    # Load model
    import numpy as np
    import yaml

    from harness.audio_io import AudioLoader
    from harness.registry import ModelRegistry

    config_path = Path(f"models/{model_id}/config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {"model_type": model_id}

    # Detect device
    import torch

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Loading model on {device}...")
    bundle = ModelRegistry.load_model(model_id, config, device)
    print(f"✓ Loaded (capabilities: {bundle['capabilities']})")

    # Load audio
    sample_rate = config.get("audio", {}).get("sample_rate", 16000)
    loader = AudioLoader(target_sample_rate=sample_rate)
    audio, sr, metadata = loader.load_audio(audio_path, model_id)

    # Ensure float32 for MPS compatibility
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    audio = np.asarray(audio, dtype=np.float32)

    print(f"✓ Audio: {metadata['duration_seconds']:.1f}s @ {sr}Hz")

    # Run task
    import time

    start = time.time()

    if task == "asr":
        if "asr" not in bundle:
            return {"error": f"Model {model_id} does not have ASR capability"}
        result = bundle["asr"]["transcribe"](audio, sr=sr, language="en")
        elapsed = time.time() - start

        text = result.get("text", "")
        print(f"\n📝 TRANSCRIPTION ({elapsed:.1f}s):\n")
        print("-" * 40)
        print(text[:500] + ("..." if len(text) > 500 else ""))
        print("-" * 40)

        return {
            "task": "asr",
            "model": model_id,
            "text": text,
            "segments": result.get("segments", []),
            "duration_s": metadata["duration_seconds"],
            "latency_s": elapsed,
            "rtf": elapsed / metadata["duration_seconds"],
        }

    elif task == "diarization":
        if "diarization" not in bundle:
            return {"error": f"Model {model_id} does not have diarization capability"}
        result = bundle["diarization"]["diarize"](audio, sr=sr)
        elapsed = time.time() - start

        turns = result.get("turns", [])
        speakers = {t.get("speaker") for t in turns}

        print(f"\n🎤 DIARIZATION ({elapsed:.1f}s):\n")
        print(f"Speakers detected: {len(speakers)}")
        print("-" * 40)
        for turn in turns[:10]:
            print(
                f"  [{turn.get('start', 0):.1f}s - {turn.get('end', 0):.1f}s] {turn.get('speaker', '?')}"
            )
        if len(turns) > 10:
            print(f"  ... and {len(turns) - 10} more turns")
        print("-" * 40)

        return {
            "task": "diarization",
            "model": model_id,
            "speakers": list(speakers),
            "num_speakers": len(speakers),
            "turns": turns,
            "duration_s": metadata["duration_seconds"],
            "latency_s": elapsed,
        }

    elif task == "vad":
        if "vad" not in bundle:
            return {"error": f"Model {model_id} does not have VAD capability"}
        result = bundle["vad"]["detect"](audio, sr=sr)
        elapsed = time.time() - start

        segments = result.get("segments", [])

        print(f"\n🔊 VAD ({elapsed:.1f}s):\n")
        print(f"Speech segments: {len(segments)}")
        print("-" * 40)
        for seg in segments[:10]:
            start_s = seg.get("start", 0)
            end_s = seg.get("end", 0)
            # Normalize if in samples
            if start_s > metadata["duration_seconds"] * 10:
                start_s = start_s / sr
                end_s = end_s / sr
            print(f"  [{start_s:.1f}s - {end_s:.1f}s]")
        print("-" * 40)

        return {
            "task": "vad",
            "model": model_id,
            "segments": segments,
            "total_duration_s": metadata["duration_seconds"],
            "latency_s": elapsed,
        }

    elif task == "tts":
        prompt = kwargs.get("prompt", "Hello, this is a test.")
        if "tts" not in bundle:
            return {"error": f"Model {model_id} does not have TTS capability"}
        result = bundle["tts"]["synthesize"](prompt)
        elapsed = time.time() - start

        output_audio = result.get("audio")
        output_sr = result.get("sr", 24000)
        duration = result.get("duration_s", 0) or (
            len(output_audio) / output_sr if output_audio is not None else 0
        )

        print(f"\n🔈 TTS ({elapsed:.1f}s):\n")
        print(f'Input: "{prompt[:50]}{"..." if len(prompt) > 50 else ""}"')
        print(f"Output: {duration:.1f}s audio")

        return {
            "task": "tts",
            "model": model_id,
            "prompt": prompt,
            "duration_s": duration,
            "latency_s": elapsed,
        }

    elif task == "v2v":
        prompt = kwargs.get("prompt", "Respond to this")
        if "v2v" not in bundle:
            return {"error": f"Model {model_id} does not have V2V capability"}
        result = bundle["v2v"]["run_v2v_turn"](audio, sr=sr, prompt=prompt)
        elapsed = time.time() - start

        response_text = result.get("response_text", "")

        print(f"\n🎙️ V2V ({elapsed:.1f}s):\n")
        print(f'Response: "{response_text[:200]}{"..." if len(response_text) > 200 else ""}"')

        return {
            "task": "v2v",
            "model": model_id,
            "response_text": response_text,
            "has_audio": result.get("audio") is not None,
            "latency_s": elapsed,
        }

    else:
        return {"error": f"Unknown task: {task}"}


def main():
    parser = argparse.ArgumentParser(
        description="Run task with decision-engine-recommended model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Selection Priority (by OUTCOME, not evidence grade):
  1. RECOMMENDED - models that pass all criteria
  2. ACCEPTABLE - models with trade-offs
  3. BEST_EFFORT - only with --policy best_effort, explicit warning

Examples:
  python scripts/recommend_and_run.py --audio file.wav --task asr
  python scripts/recommend_and_run.py --audio file.wav --task asr --policy best_effort
  python scripts/recommend_and_run.py --use-case meeting_analysis --audio meeting.wav
        """,
    )
    parser.add_argument("--audio", type=str, help="Path to audio file")
    parser.add_argument(
        "--task",
        type=str,
        choices=["asr", "diarization", "vad", "tts", "v2v"],
        help="Task to perform",
    )
    parser.add_argument("--use-case", type=str, dest="use_case", help="Use case (runs pipeline)")
    parser.add_argument("--model", type=str, default=None, help="Override model selection")
    parser.add_argument(
        "--policy",
        type=str,
        default="strict",
        choices=["strict", "best_effort"],
        help="Selection policy (default: strict)",
    )
    parser.add_argument(
        "--include-in-arsenal",
        action="store_true",
        help="Include run in arsenal (default: excluded)",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for TTS/V2V")

    args = parser.parse_args()

    # Validate
    if not args.task and not args.use_case:
        parser.error("Either --task or --use-case is required")

    if args.task != "tts" and not args.audio:
        parser.error("--audio is required for non-TTS tasks")

    audio_path = Path(args.audio) if args.audio else None
    if audio_path and not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return 1

    # Use case pipeline mode
    if args.use_case:
        pipeline, outcome, reason, is_best_effort = get_pipeline_for_use_case(
            args.use_case, args.policy
        )

        if not pipeline:
            print(f"❌ {reason}")
            return 1

        print(f"\n🎯 Use Case: {args.use_case}")
        print(f"   Outcome: {outcome}")
        print(f"   Pipeline: {pipeline}")
        if is_best_effort:
            print("   ⚠️  BEST_EFFORT MODE - Results not evidence-quality")
        print()

        # Run each task in pipeline
        for task, model_id in pipeline.items():
            if task in ["primary", "secondary"]:  # Skip metadata keys
                continue
            try:
                result = run_task(task, model_id, audio_path, prompt=args.prompt)
                if "error" in result:
                    print(f"❌ {task}: {result['error']}")
            except Exception as e:
                print(f"❌ {task}: {e}")

        return 0

    # Single task mode
    if args.model:
        model_id = args.model
        outcome = "OVERRIDE"
        reason = f"User-specified: {model_id}"
        is_best_effort = False
    else:
        model_id, outcome, reason, is_best_effort = get_best_model_for_task(args.task, args.policy)

    if not model_id:
        print(f"❌ {reason}")
        if args.policy == "strict":
            print("   Hint: Use --policy best_effort to try available models anyway")
        return 1

    print(f"\n🎯 {reason}")
    if is_best_effort:
        print("   ⚠️  BEST_EFFORT MODE - Results excluded from arsenal by default")

    # Run
    try:
        result = run_task(args.task, model_id, audio_path, prompt=args.prompt)

        if "error" in result:
            print(f"\n❌ {result['error']}")
            return 1

        print(f"\n✅ Done! RTF: {result.get('rtf', result.get('latency_s', 0)):.2f}x")

        # Save result
        if is_best_effort and not args.include_in_arsenal:
            output_dir = Path(f"runs_best_effort/{model_id}/{args.task}")
        else:
            output_dir = Path(f"runs/{model_id}/{args.task}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Mark if excluded from arsenal
        result["_meta"] = {
            "outcome": outcome,
            "is_best_effort": is_best_effort,
            "decision_excluded": is_best_effort and not args.include_in_arsenal,
            "timestamp": datetime.now().isoformat(),
        }

        output_file = output_dir / f"adhoc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

        if is_best_effort and not args.include_in_arsenal:
            print(f"📄 Result saved to: {output_file} (excluded from arsenal)")
        else:
            print(f"📄 Result saved to: {output_file}")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
