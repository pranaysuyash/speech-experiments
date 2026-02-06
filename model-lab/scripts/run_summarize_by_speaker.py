#!/usr/bin/env python3
"""
Summarize By Speaker - Generate PER-SPEAKER summaries using alignment.

Pipeline:
1. Run/Load Alignment artifact (ASR + Diarization + Alignment)
2. Group aligned segments by speaker
3. Summarize each speaker's contribution independently
4. Aggregate into a meeting overview

Usage:
    python scripts/run_summarize_by_speaker.py --input meeting.mp4
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.alignment import AlignedSegment, AlignedTranscript
from harness.llm_provider import get_llm_completion
from harness.nlp_schema import (
    NLPArtifact,
    NLPInputs,
    NLPProvenance,
    NLPRunContext,
    compute_file_hash,
    compute_text_hash,
)

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_summarize_by_speaker")

DEFAULT_NLP_MODEL = "gemini-2.0-flash"

SPEAKER_SUMMARY_PROMPT = """Summarize what {speaker} said/did in this meeting.
Focus on their key arguments, decisions, and action items.
Do not attribute other speakers' points to them.
Keep it concise (aim for < 5 bullets).

Transcript of {speaker}:
{transcript}

Summary (bullet points):"""


def run_alignment_dependency(input_path: Path, pre: str | None = None) -> Path:
    """Run alignment pipeline."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_alignment.py"),
        "--input",
        str(input_path),
    ]
    if pre:
        cmd.extend(["--pre", pre])

    logger.info("Running alignment pipeline...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Alignment failed: {result.stderr[-500:]}")

    for line in result.stdout.split("\n"):
        if line.startswith("ARTIFACT_PATH:"):
            return Path(line.split(":", 1)[1].strip())

    raise RuntimeError("Alignment produced no artifact path")


def load_alignment_artifact(path: Path) -> AlignedTranscript:
    with open(path) as f:
        data = json.load(f)

    # Reconstruct objects (simplified)
    # Ideally use from_dict but AlignedTranscript is simple enough
    segments = [AlignedSegment(**s) for s in data["output"]["segments"]]

    # Mock metrics/paths for now since we just need segments
    from harness.alignment import AlignmentMetrics

    metrics = AlignmentMetrics(**data["output"]["metrics"])

    return AlignedTranscript(
        segments=segments,
        metrics=metrics,
        source_asr_path=data["output"]["source_asr_path"],
        source_diarization_path=data["output"]["source_diarization_path"],
    )


def summarize_speaker_content(speaker: str, text: str, nlp_model: str) -> list[str]:
    """Summarize a single speaker's text."""
    if len(text.split()) < 20:
        return []  # Too short to summarize

    prompt = SPEAKER_SUMMARY_PROMPT.format(speaker=speaker, transcript=text[:15000])
    prompt_hash = compute_text_hash(prompt)
    text_hash = compute_text_hash(text)

    result = get_llm_completion(
        prompt=prompt,
        model=nlp_model,
        text_hash=text_hash,
        prompt_hash=prompt_hash,
        use_cache=True,
    )

    if result.success and result.text:
        lines = result.text.strip().split("\n")
        bullets = [l.strip().lstrip("-•*").strip() for l in lines if len(l.strip()) > 5]
        return bullets
    return []


def run_summarize_by_speaker(
    alignment_artifact_path: Path,
    nlp_model: str = DEFAULT_NLP_MODEL,
) -> tuple[dict[str, Any], Path]:
    logger.info(f"Loading alignment: {alignment_artifact_path.name}")
    aligned_transcript = load_alignment_artifact(alignment_artifact_path)

    # Group by speaker
    speaker_text = defaultdict(list)
    for seg in aligned_transcript.segments:
        if seg.speaker_id != "unknown":
            speaker_text[seg.speaker_id].append(seg.text)

    logger.info(f"Speakers found: {list(speaker_text.keys())}")

    speaker_summaries = {}

    for speaker, texts in speaker_text.items():
        full_text = " ".join(texts)
        logger.info(f"Summarizing {speaker} ({len(full_text.split())} words)...")

        bullets = summarize_speaker_content(speaker, full_text, nlp_model)
        if bullets:
            speaker_summaries[speaker] = bullets

    # Create artifact
    # Schema: We'll misuse SummaryOutput slightly or extend it?
    # Let's just create a custom output dict, verifying basics

    output = {
        "speaker_summaries": speaker_summaries,
        "speaker_stats": aligned_transcript.metrics.to_dict(),
    }

    # Provenance
    run_context = NLPRunContext(
        task="summarize_by_speaker",
        nlp_model_id=nlp_model,
        timestamp=datetime.now().isoformat(),
        command=sys.argv,
    )

    inputs = NLPInputs(
        parent_artifact_path=str(alignment_artifact_path),
        parent_artifact_hash=compute_file_hash(alignment_artifact_path),
        asr_model_id="alignment",
        asr_text_hash=compute_file_hash(alignment_artifact_path),  # Proxy
        transcript_word_count=len(aligned_transcript.segments),
        audio_duration_s=aligned_transcript.metrics.total_duration_s,
    )

    artifact = NLPArtifact(
        run_context=run_context,
        inputs=inputs,
        provenance=NLPProvenance(
            prompt_template=SPEAKER_SUMMARY_PROMPT[:100] + "...",
            prompt_hash=compute_text_hash(SPEAKER_SUMMARY_PROMPT),
        ),
        output=output,
        metrics_structural={},
        gates={"has_failure": False},
    )

    # Save
    runs_dir = Path("runs/nlp/summarize_by_speaker")
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_file = runs_dir / f"summary_by_speaker_{int(time.time())}.json"

    with open(run_file, "w") as f:
        json.dump(artifact.to_dict(), f, indent=2)

    logger.info(f"✓ Artifact saved: {run_file}")
    print(f"ARTIFACT_PATH:{run_file}")

    return artifact.to_dict(), run_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="Input audio/video")
    parser.add_argument("--from-artifact", type=Path, help="Alignment artifact")
    parser.add_argument("--nlp-model", default=DEFAULT_NLP_MODEL)
    parser.add_argument("--pre", help="Preprocessing operators")

    args = parser.parse_args()

    try:
        if args.from_artifact:
            alignment_path = args.from_artifact
        elif args.input:
            alignment_path = run_alignment_dependency(args.input, args.pre)
        else:
            raise ValueError("Must provide --input or --from-artifact")

        result, path = run_summarize_by_speaker(alignment_path, args.nlp_model)

        print("\n--- Speaker Summaries ---")
        for speaker, bullets in result["output"]["speaker_summaries"].items():
            print(f"\n👤 {speaker}:")
            for b in bullets:
                print(f"  - {b}")

    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
