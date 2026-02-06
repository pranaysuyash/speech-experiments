#!/usr/bin/env python3
"""
Run Speaker Alignment - Align ASR transcript with Diarization turns.

Pipeline:
1. Run ASR (if needed)
2. Run Diarization (if needed)
3. Align segments using temporal overlap
4. Produce AlignedTranscript artifact

Usage:
    python scripts/run_alignment.py --input meeting.mp4
    python scripts/run_alignment.py --asr-artifact ... --diarization-artifact ...
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.alignment import align_artifacts
from harness.nlp_schema import (
    NLPArtifact,
    NLPInputs,
    NLPProvenance,
    NLPRunContext,
    compute_file_hash,
)

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_alignment")


def run_dependency(script_name: str, input_path: Path, args_list: list = None) -> Path:
    """Run a dependency script and return its artifact path."""
    if args_list is None:
        args_list = []
    cmd = [
        sys.executable,
        str(Path(__file__).parent / script_name),
        "--input",
        str(input_path),
    ] + args_list
    logger.info(f"Running {script_name}...")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed: {result.stderr[-500:]}")

    for line in result.stdout.split("\n"):
        if line.startswith("ARTIFACT_PATH:"):
            return Path(line.split(":", 1)[1].strip())

    raise RuntimeError(f"{script_name} produced no artifact path")


def run_alignment_pipeline(
    input_path: Path | None = None,
    asr_artifact_path: Path | None = None,
    diarization_artifact_path: Path | None = None,
    pre: str | None = None,
) -> tuple[dict[str, Any], Path]:
    # 1. Resolve dependencies
    if input_path:
        if not asr_artifact_path:
            # Use model_app recommend for ASR to get best model
            logger.info("Resolving ASR...")
            asr_artifact_path = run_dependency(
                "model_app.py",
                input_path,
                ["recommend", "--task", "asr", "--audio", str(input_path)]
                + (["--pre", pre] if pre else []),
            )

        if not diarization_artifact_path:
            # Use model_app recommend for Diarization
            logger.info("Resolving Diarization...")
            diar_args = ["recommend", "--task", "diarization", "--audio", str(input_path)]
            if pre:
                diar_args.extend(["--pre", pre])
            diarization_artifact_path = run_dependency("model_app.py", input_path, diar_args)

    if not asr_artifact_path or not diarization_artifact_path:
        raise ValueError("Must provide --input or both artifacts")

    if not asr_artifact_path.exists():
        raise FileNotFoundError(f"ASR artifact not found: {asr_artifact_path}")
    if not diarization_artifact_path.exists():
        raise FileNotFoundError(f"Diarization artifact not found: {diarization_artifact_path}")

    # 2. Perform Alignment
    logger.info(f"Aligning {asr_artifact_path.name} with {diarization_artifact_path.name}...")
    alignment = align_artifacts(str(asr_artifact_path), str(diarization_artifact_path))

    # 3. Create Artifact
    run_context = NLPRunContext(
        task="alignment",
        nlp_model_id="alignment_v1",  # Deterministic logic
        timestamp=datetime.now().isoformat(),
        command=sys.argv,
    )

    inputs = NLPInputs(
        parent_artifact_path=str(asr_artifact_path),
        parent_artifact_hash=compute_file_hash(asr_artifact_path),
        # Store diarization artifact as secondary parent?
        # For now, just track it in provenance or inputs
        asr_model_id="alignment",  # Not really applicable
        asr_text_hash=compute_file_hash(
            diarization_artifact_path
        ),  # Hacking this field to store diarization hash?
        transcript_word_count=len(alignment.segments),
        audio_duration_s=alignment.metrics.total_duration_s,
    )

    # We need to extend InputsSchema or use metadata for diarization linkage
    # For now, I'll store it in extra fields since NLPInputs is strict
    # Actually, let's just make sure we capture it in the output structure which is flexible

    artifact = NLPArtifact(
        run_context=run_context,
        inputs=inputs,
        provenance=NLPProvenance(
            prompt_template="deterministic_overlap",
            prompt_hash="v1",
        ),
        output=alignment.to_dict(),
        metrics_structural=alignment.metrics.to_dict(),
        gates={"has_failure": False},
    )

    # Save
    runs_dir = Path("runs/alignment")
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_file = runs_dir / f"alignment_{int(time.time())}.json"

    with open(run_file, "w") as f:
        json.dump(artifact.to_dict(), f, indent=2)

    logger.info(f"✓ Alignment artifact saved: {run_file}")
    logger.info(f"  Coverage: {alignment.metrics.coverage_ratio:.1%}")
    logger.info(f"  Speakers: {len(alignment.metrics.speaker_distribution)}")

    print(f"ARTIFACT_PATH:{run_file}")
    return artifact.to_dict(), run_file


def main():
    parser = argparse.ArgumentParser(description="Run speaker alignment")
    parser.add_argument("--input", type=Path, help="Input audio/video")
    parser.add_argument("--asr-artifact", type=Path, help="Pre-computed ASR artifact")
    parser.add_argument(
        "--diarization-artifact", type=Path, help="Pre-computed Diarization artifact"
    )
    parser.add_argument("--pre", help="Preprocessing operators")

    args = parser.parse_args()

    try:
        run_alignment_pipeline(
            input_path=args.input,
            asr_artifact_path=args.asr_artifact,
            diarization_artifact_path=args.diarization_artifact,
            pre=args.pre,
        )
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
