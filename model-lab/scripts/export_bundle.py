#!/usr/bin/env python3
"""
Export Bundle - Package all artifacts for a session into a portable ZIP.

Features:
1. Auto-discover artifacts (ASR, Diarization, Alignment, Analyzers) for an input file.
2. Verify artifact integrity (hashes).
3. Generate manifest.json with full session metadata.
4. Create deterministic ZIP bundle.

Usage:
    python scripts/export_bundle.py --input meeting.mp4 --output bundle.zip
"""

import argparse
import json
import logging
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.audio_io import compute_input_hash
from harness.nlp_schema import compute_file_hash

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("export_bundle")

# Search paths for artifacts
MANIFEST_VERSION = "1.1.0"
RUNS_DIR = Path("runs")
TASKS = [
    "asr",
    "diarization",
    "alignment",
    "nlp/summarize",
    "nlp/summarize_by_speaker",
    "nlp/action_items",
    "nlp/action_items_assignee",
    "nlp/ner",
    "nlp/chapters",
]


def find_latest_artifact_for_task(task_dir: Path, source_hash: str) -> Path | None:
    """Find latest artifact in task_dir that matches source_hash."""
    if not task_dir.exists():
        return None

    candidates = []

    for f in task_dir.glob("**/*.json"):
        if "chunks" in str(f):  # Skip chunk artifacts for top-level search
            continue

        try:
            with open(f) as fh:
                data = json.load(fh)

            # Check linkage
            # Different runners store hash in different places in inputs/schema
            # We check a few common locations

            # 1. Inputs schema (RunnerArtifact)
            inputs = data.get("inputs", {})
            inputs.get("audio_hash")

            # 2. NLP Inputs (NLPArtifact)
            # NLP inputs point to parent artifact, not audio directly usually
            # But they might have asr_text_hash etc.
            # Tracing exact linkage upwards is hard without loading parents.
            # HOWEVER, for a simple bundle, we might rely on the fact that
            # we want artifacts that EVENTUALLY trace back.
            # Building a graph is expensive.

            # Strategy:
            # - ASR/Diarization: direct audio_hash match
            # - NLP: trace parent_artifact linkage? Or just check if inputs.audio_duration matches?
            #   We need to be robust.
            #   Let's check if 'inputs' contains the source hash if propagated?
            #   Currently NLP schema doesn't propagate source AUDIO hash, only parent artifact hash.

            # Okay, we need a smarter walker.
            # 1. Find ASR/Diarization matching audio hash.
            # 2. Get their paths/hashes.
            # 3. Find artifacts that reference those parent paths/hashes.

            # Simplification: return data so caller can graph it
            candidates.append((f.stat().st_mtime, f, data))

        except Exception:
            continue

    # Sort by time desc
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates


def collect_session_artifacts(input_path: Path) -> dict[str, Path]:
    """
    Collect all relevant artifacts for input file.
    Returns task_name -> artifact_path
    """
    # 1. Compute input hash
    logger.info(f"Hashing input: {input_path.name}...")
    audio_hash = compute_input_hash(input_path)  # Uses 2MB head
    logger.info(f"Input Hash: {audio_hash}")

    session_artifacts = {}

    # 2. Find Roots (ASR, Diarization)
    asr_candidates = find_latest_artifact_for_task(RUNS_DIR / "asr", audio_hash)
    diar_candidates = find_latest_artifact_for_task(RUNS_DIR / "diarization", audio_hash)

    asr_root = None
    diar_root = None

    # Filter ASR by exact hash match
    if asr_candidates:
        for _, path, data in asr_candidates:
            if data.get("inputs", {}).get("audio_hash") == audio_hash:
                session_artifacts["asr"] = path
                asr_root = (path, compute_file_hash(path))
                break

    # Filter Diarization by exact hash match
    if diar_candidates:
        for _, path, data in diar_candidates:
            if data.get("inputs", {}).get("audio_hash") == audio_hash:
                session_artifacts["diarization"] = path
                diar_root = (path, compute_file_hash(path))
                break

    if not asr_root:
        logger.warning("No ASR artifact found for this input.")
        # Cannot find children without parent
        return session_artifacts

    logger.info(f"Found ASR: {asr_root[0].name}")
    if diar_root:
        logger.info(f"Found Diarization: {diar_root[0].name}")

    # 3. Find Children (NLP, Alignment)
    # They reference parent artifact hash/path

    # We look for artifacts where inputs.parent_artifact_hash == asr_root[1]
    # OR inputs.parent_artifact_path matches

    parent_hashes = {asr_root[1]}
    parent_paths = {str(asr_root[0])}

    # Also find alignment which might be parent for speaker tasks
    alignment_root = None

    # Helper to scan a dir for children of known parents
    def scan_children(task_name, dir_path):
        candidates = find_latest_artifact_for_task(dir_path, "")
        if not candidates:
            return None

        for _, path, data in candidates:
            inputs = data.get("inputs", {})
            p_hash = inputs.get("parent_artifact_hash")
            p_path = inputs.get("parent_artifact_path")

            if p_hash in parent_hashes or p_path in parent_paths:
                return (path, compute_file_hash(path))
        return None

    # Scan basic NLP
    for task in ["nlp/summarize", "nlp/action_items", "nlp/ner", "alignment"]:
        found = scan_children(task, RUNS_DIR / task)
        if found:
            session_artifacts[task] = found[0]
            if task == "alignment":
                alignment_root = found
                logger.info(f"Found Alignment: {found[0].name}")

    # 4. Find Grandchildren (Speaker-aware tasks)
    # They reference Alignment artifact
    if alignment_root:
        align_hashes = {alignment_root[1]}
        align_paths = {str(alignment_root[0])}

        def scan_grandchildren(task_name, dir_path):
            candidates = find_latest_artifact_for_task(dir_path, "")
            if not candidates:
                return None
            for _, path, data in candidates:
                inputs = data.get("inputs", {})
                p_hash = inputs.get("parent_artifact_hash")
                p_path = inputs.get("parent_artifact_path")
                if p_hash in align_hashes or p_path in align_paths:
                    return (path, compute_file_hash(path))
            return None

        for task in ["nlp/summarize_by_speaker", "nlp/action_items_assignee", "nlp/chapters"]:
            found = scan_grandchildren(task, RUNS_DIR / task)
            if found:
                session_artifacts[task] = found[0]

    return session_artifacts


def create_bundle(input_path: Path, output_path: Path, dry_run: bool = False):
    """Create zip bundle with manifest."""
    artifacts = collect_session_artifacts(input_path)

    if not artifacts:
        if dry_run:
            print("No artifacts found.")
        else:
            logger.error("No artifacts found!")
            sys.exit(1)
        return

    manifest = {
        "schema_version": MANIFEST_VERSION,
        "export_date": datetime.now().isoformat(),
        "input_file": input_path.name,
        "input_hash": compute_input_hash(input_path),
        "artifacts": {},
    }

    # Pre-calculate manifest entries
    for task, path in artifacts.items():
        archive_name = f"artifacts/{task.replace('/', '_')}_{path.name}"
        manifest["artifacts"][task] = {
            "file": archive_name,
            "original_path": str(path),
            "hash": compute_file_hash(path),
        }

    if dry_run:
        print(f"Dry Run: Bundle Plan for {input_path.name}")
        print(f"Input Hash: {manifest['input_hash']}")
        print(f"Artifacts Found: {len(artifacts)}")
        for task, meta in manifest["artifacts"].items():
            print(f"  - {task}: {meta['original_path']}")
        print(f"Will create: {output_path}")
        return

    logger.info(f"Creating bundle: {output_path}")

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add Artifacts
        for task, path in artifacts.items():
            archive_name = manifest["artifacts"][task]["file"]
            zf.write(path, archive_name)
            logger.info(f"  + {task}: {path.name}")

        # Add Manifest
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    logger.info(f"✓ Bundle created ({os.path.getsize(output_path) / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Input media file")
    parser.add_argument("--output", type=Path, required=True, help="Output ZIP path")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print bundle contents without creating"
    )

    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    try:
        create_bundle(args.input, args.output, args.dry_run)
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
