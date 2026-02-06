#!/usr/bin/env python3
"""
Run Chapters Extraction - Create semantic table of contents.

Pipeline:
1. Load Aligned Transcript (or ASR)
2. Semantic Segmentation (Embeddings + Cosine Sim)
3. LLM Enrichment (Titles + Summaries)
4. Save chapters.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.alignment import load_alignment
from harness.chapters import Chapter, SemanticSegmenter
from harness.llm_provider import get_llm_completion
from harness.nlp_schema import compute_file_hash

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_chapters")

CHAPTER_PROMPT = """You are an editor creating a Table of Contents for a meeting transcript.
Analyze the following section of text and provide:
1. A concise Title (max 10 words).
2. A short Summary (1-2 sentences).

TEXT:
{text}

Respond in JSON:
{{
  "title": "...",
  "summary": "..."
}}
"""


def enrich_chapter(chapter: Chapter, context: str):
    """Enrich chapter with Title/Summary from LLM."""
    # Limit context to avoid token limits (though chapters are usually short)
    # 60-300s of text is usually few hundred tokens. Safe.

    prompt = CHAPTER_PROMPT.format(text=context[:5000])  # Safe limit

    result = get_llm_completion(prompt)
    if result.success:
        try:
            # Clean possible markdown code blocks
            clean_text = result.text.strip().replace("```json", "").replace("```", "")
            data = json.loads(clean_text)
            chapter.title = data.get("title", chapter.title)
            chapter.summary = data.get("summary", chapter.summary)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response for chapter {chapter.index}")
    else:
        logger.warning(f"LLM failure for chapter {chapter.index}: {result.error_message}")

    # 2. Segment
    logger.info("Running semantic segmentation...")

    (
        Path("runs/cache/embeddings")
        if not os.environ.get("EMBEDDING_CACHE_DIR")
        else Path(os.environ.get("EMBEDDING_CACHE_DIR"))
    )
    # Allow override via args if I added them to run_chapters args...
    # Let's add them to the main() parser and pass them here run_chapters(..., cache_dir=...).
    # But run_chapters signature currently is just alignment_path.

    # Let's update run_chapters signature first? Or just use env/default here for now as quick fix?
    # User wanted explicit args. I should update signature.

    pass


def run_chapters(alignment_path: Path, cache_dir: Path | None = None):
    """Run extraction pipeline."""
    # 1. Load Data
    if not alignment_path.exists():
        logger.error(f"Alignment not found: {alignment_path}")
        sys.exit(1)

    logger.info(f"Loading alignment: {alignment_path}")
    aligned_transcript = load_alignment(alignment_path)

    # 2. Segment
    logger.info("Running semantic segmentation...")
    if cache_dir:
        logger.info(f"Embedding cache: {cache_dir}")
        cache_dir.mkdir(parents=True, exist_ok=True)

    segmenter = SemanticSegmenter(cache_dir=cache_dir)

    if aligned_transcript.segments:
        duration = aligned_transcript.segments[-1].end_s + 2.0  # Buffer
    else:
        logger.warning("Empty transcript!")
        return

    chapters, config = segmenter.segment(aligned_transcript.segments, duration)
    logger.info(f"Identified {len(chapters)} chapters.")

    # 3. Enrich (LLM)
    logger.info("Enriching chapters with Titles/Summaries...")
    for chap in chapters:
        texts = [s["text"] for s in chap.evidence_segments]
        full_text = " ".join(texts)
        if not full_text.strip():
            continue

        enrich_chapter(chap, full_text)
        logger.info(f"  [{chap.index}] {chap.title} ({chap.start:.1f}-{chap.end:.1f}s)")

    # 4. Save Artifact
    input_hash = compute_file_hash(alignment_path)

    output_data = {
        "chapters": [c.to_dict() for c in chapters],
        "stats": {
            "count": len(chapters),
            "avg_duration": sum(c.end - c.start for c in chapters) / len(chapters)
            if chapters
            else 0,
        },
        "config": config,
        "inputs": {"parent_artifact_path": str(alignment_path), "parent_artifact_hash": input_hash},
    }

    output_dir = Path("runs/nlp/chapters")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"chapters_{alignment_path.stem}.json"
    output_path = output_dir / output_filename

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved chapters to {output_path}")
    print(f"ARTIFACT_PATH: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="Input media file")
    parser.add_argument("--from-artifact", type=Path, help="Direct path to alignment.json")
    parser.add_argument("--pre", help="Preprocessing flags (passed to upstream)")
    parser.add_argument(
        "--embedding-cache-dir", type=Path, help="Cache directory for embedding models"
    )

    args = parser.parse_args()

    try:
        # Default cache in repo if not provided
        cache_dir = args.embedding_cache_dir
        if not cache_dir:
            # harness/../runs/cache
            # Repo root logic:
            repo_root = Path(__file__).parent.parent
            cache_dir = repo_root / "runs/cache/embeddings"

        if args.from_artifact:
            path = args.from_artifact
        elif args.input:
            from scripts.run_alignment import run_alignment_pipeline

            logger.info("Resolving upstream dependency (Alignment)...")
            artifact = run_alignment_pipeline(str(args.input), force=False)
            path = Path(artifact.path)
        else:
            logger.error("Must provide --input or --from-artifact")
            sys.exit(1)

        run_chapters(path, cache_dir=cache_dir)

    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
