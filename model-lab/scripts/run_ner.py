#!/usr/bin/env python3
"""
NER From ASR - Named Entity Recognition from transcript.

Hybrid extraction:
1. Regex (deterministic): EMAIL, PHONE, URL, MONEY, DATE, TIME
2. LLM (per-chunk): PERSON, ORG, LOC, PRODUCT

Entities are deduplicated across chunks by normalized text + type.
"""

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.chunking import ChunkingPolicy, chunk_transcript
from harness.llm_provider import get_llm_completion
from harness.nlp_schema import (
    ENTITY_TYPES,
    NER_PROMPT_TEMPLATE,
    Entity,
    NEROutput,
    NLPArtifact,
    NLPInputs,
    NLPProvenance,
    NLPRunContext,
    compute_file_hash,
    compute_text_hash,
    load_asr_artifact,
    validate_nlp_artifact,
)
from harness.transcript_view import from_asr_artifact

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_ner")

DEFAULT_NLP_MODEL = "gemini-2.0-flash"
MAX_ENTITIES = 200


# ============ REGEX PATTERNS (deterministic) ============

REGEX_PATTERNS = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "PHONE": r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
    "URL": r'https?://[^\s<>"{}|\\^`\[\]]+',
    "MONEY": r"\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b|\b\d+(?:\.\d+)?\s*(?:dollars?|USD|EUR|GBP|cents?)\b",
    "DATE": r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?\b|\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b",
    "TIME": r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b",
}


def extract_regex_entities(text: str, chunk_id: int = 0) -> list[dict]:
    """Extract entities using regex patterns (deterministic)."""
    entities = []

    for entity_type, pattern in REGEX_PATTERNS.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            entities.append(
                {
                    "text": match.group().strip(),
                    "type": entity_type,
                    "confidence": 1.0,
                    "source_chunk_id": chunk_id,
                    "method": "regex",
                }
            )

    return entities


# ============ LLM EXTRACTION ============


def parse_ner_response(text: str) -> list[dict]:
    """Parse LLM JSON response into entity list."""
    text = text.strip()

    json_match = re.search(r"\{[\s\S]*\}", text)
    if not json_match:
        return []

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return []

    entities = data.get("entities", [])
    if not isinstance(entities, list):
        return []

    result = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        text_val = e.get("text", "")
        type_val = e.get("type", "").upper()

        if text_val and type_val in ENTITY_TYPES:
            result.append(
                {
                    "text": text_val.strip(),
                    "type": type_val,
                    "confidence": 0.8,  # LLM extraction confidence
                    "method": "llm",
                }
            )

    return result


def extract_llm_entities(
    text: str,
    text_hash: str,
    nlp_model: str,
    chunk_id: int = 0,
) -> tuple[list[dict], bool]:
    """Extract PERSON/ORG/LOC/PRODUCT using LLM."""
    prompt_hash = compute_text_hash(NER_PROMPT_TEMPLATE)

    prompt = NER_PROMPT_TEMPLATE.format(transcript=text[:15000])

    result = get_llm_completion(
        prompt=prompt,
        model=nlp_model,
        text_hash=text_hash,
        prompt_hash=prompt_hash,
        max_attempts=3,
        use_cache=True,
    )

    if result.success:
        entities = parse_ner_response(result.text)
        for e in entities:
            e["source_chunk_id"] = chunk_id
        return entities, True
    else:
        return [], False


# ============ DEDUP AND MERGE ============


def normalize_entity_text(text: str) -> str:
    """Normalize entity text for deduplication."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def merge_entities(all_entities: list[dict]) -> list[Entity]:
    """
    Merge entities across chunks, collapsing duplicates.

    Key: (normalized_text, type)
    Merged: count, source_chunk_ids
    """
    merged = {}

    for e in all_entities:
        key = (normalize_entity_text(e["text"]), e["type"])

        if key not in merged:
            merged[key] = {
                "text": e["text"],  # Keep first occurrence's casing
                "type": e["type"],
                "count": 0,
                "source_chunk_ids": set(),
                "confidence": e.get("confidence", 1.0),
            }

        merged[key]["count"] += 1
        merged[key]["source_chunk_ids"].add(e.get("source_chunk_id", 0))
        # Keep highest confidence
        merged[key]["confidence"] = max(merged[key]["confidence"], e.get("confidence", 1.0))

    # Convert to Entity objects
    result = []
    for data in merged.values():
        result.append(
            Entity(
                text=data["text"],
                type=data["type"],
                count=data["count"],
                source_chunk_ids=sorted(data["source_chunk_ids"]),
                confidence=data["confidence"],
            )
        )

    # Sort by count (most frequent first), then alphabetically
    result.sort(key=lambda e: (-e.count, e.text.lower()))

    return result


# ============ MAIN RUNNER ============


def run_ner(
    asr_artifact_path: Path,
    nlp_model: str = DEFAULT_NLP_MODEL,
    chunking_policy: ChunkingPolicy | None = None,
) -> tuple[dict[str, Any], Path]:
    """
    Extract named entities from ASR artifact.

    Hybrid approach:
    1. Regex globally for EMAIL/PHONE/URL/MONEY/DATE/TIME
    2. LLM per-chunk for PERSON/ORG/LOC/PRODUCT
    3. Merge and dedupe across all sources
    """
    logger.info(f"=== NER from ASR: {asr_artifact_path.name} ===")

    # Load ASR artifact
    asr_artifact = load_asr_artifact(asr_artifact_path)
    asr_artifact_hash = compute_file_hash(asr_artifact_path)
    asr_model_id = asr_artifact["run_context"]["model_id"]
    audio_duration = asr_artifact["inputs"].get("audio_duration_s", 0)

    # Create transcript view
    view = from_asr_artifact(asr_artifact_path)
    logger.info(f"Transcript: {view.word_count} words")

    # Global regex extraction (deterministic)
    logger.info("Extracting regex entities (EMAIL/PHONE/URL/MONEY/DATE/TIME)...")
    regex_entities = extract_regex_entities(view.full_text, chunk_id=0)
    logger.info(f"  Found {len(regex_entities)} regex entities")

    # Chunk for LLM extraction
    if chunking_policy is None:
        chunking_policy = ChunkingPolicy()

    chunk_result = chunk_transcript(view, chunking_policy)

    # LLM extraction per chunk
    logger.info(f"Extracting LLM entities from {chunk_result.total_chunks} chunks...")
    llm_entities = []
    children = []

    for chunk in chunk_result.chunks:
        entities, success = extract_llm_entities(
            chunk.text,
            chunk.text_hash,
            nlp_model,
            chunk_id=chunk.chunk_id,
        )

        llm_entities.extend(entities)
        children.append(
            {
                "chunk_id": chunk.chunk_id,
                "text_hash": chunk.text_hash,
                "success": success,
                "entity_count": len(entities),
            }
        )

        logger.info(f"  Chunk {chunk.chunk_id}: {len(entities)} entities")

    # Merge all entities
    all_entities = regex_entities + llm_entities
    merged = merge_entities(all_entities)

    logger.info(f"Merged: {len(all_entities)} → {len(merged)} unique entities")

    # Enforce operability limit
    truncated = False
    if len(merged) > MAX_ENTITIES:
        logger.warning(f"Truncating from {len(merged)} to {MAX_ENTITIES}")
        merged = merged[:MAX_ENTITIES]
        truncated = True

    # Count by type
    type_counts: dict[str, int] = defaultdict(int)
    for e in merged:
        type_counts[e.type] += 1

    # Build output
    output = NEROutput(
        entities=merged,
        total_entities=len(merged),
        source_word_count=view.word_count,
        entity_type_counts=dict(type_counts),
    )

    violations = output.validate()

    prompt_hash = compute_text_hash(NER_PROMPT_TEMPLATE)

    artifact = NLPArtifact(
        run_context=NLPRunContext(
            task="ner",
            nlp_model_id=nlp_model,
            timestamp=datetime.now().isoformat(),
            command=sys.argv,
        ),
        inputs=NLPInputs(
            parent_artifact_path=str(asr_artifact_path),
            parent_artifact_hash=asr_artifact_hash,
            asr_model_id=asr_model_id,
            asr_text_hash=view.text_hash,
            transcript_word_count=view.word_count,
            audio_duration_s=audio_duration,
        ),
        provenance=NLPProvenance(
            prompt_template=NER_PROMPT_TEMPLATE[:100] + "...",
            prompt_hash=prompt_hash,
        ),
        output=output.to_dict(),
        metrics_structural={
            "chunk_count": chunk_result.total_chunks,
            "regex_entities": len(regex_entities),
            "llm_entities": len(llm_entities),
            "merged_entities": len(merged),
            "truncated": truncated,
        },
        gates={
            "has_failure": False,
            "has_operability_violations": len(violations) > 0,
        },
        errors=violations,
    )

    validate_nlp_artifact(artifact)

    # Add chunking and extraction info
    artifact_dict = artifact.to_dict()
    artifact_dict["chunking"] = chunk_result.policy.to_dict()
    artifact_dict["children"] = children
    artifact_dict["extraction"] = {
        "regex_types": list(REGEX_PATTERNS.keys()),
        "llm_types": ["PERSON", "ORG", "LOC", "PRODUCT"],
    }

    # Save
    runs_dir = Path("runs/nlp/ner")
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_file = runs_dir / f"ner_{int(time.time())}.json"

    with open(run_file, "w") as f:
        json.dump(artifact_dict, f, indent=2, default=str)

    logger.info(f"✓ NER artifact saved: {run_file}")
    print(f"ARTIFACT_PATH:{run_file}")

    return artifact_dict, run_file


def run_asr_first(input_path: Path, asr_model: str | None = None, pre: str | None = None) -> Path:
    """Run ASR on input file and return artifact path."""
    logger.info(f"Running ASR on {input_path.name}...")

    if asr_model:
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "run_asr.py"),
            "--model",
            asr_model,
            "--input",
            str(input_path.resolve()),
        ]
    else:
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "model_app.py"),
            "recommend",
            "--task",
            "asr",
            "--audio",
            str(input_path.resolve()),
        ]

    if pre:
        cmd.extend(["--pre", pre])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ASR failed: {result.stderr[-200:]}")

    for line in result.stdout.split("\n"):
        if line.startswith("ARTIFACT_PATH:"):
            return Path(line.split(":", 1)[1].strip())

    raise RuntimeError("ASR did not produce artifact path")


def main():
    parser = argparse.ArgumentParser(description="Extract named entities from ASR transcript")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--from-artifact", type=Path, help="ASR artifact path")
    input_group.add_argument("--input", type=Path, help="Audio/video file")

    parser.add_argument("--asr-model", help="ASR model (convenience mode)")
    parser.add_argument("--pre", help="Preprocessing operators (convenience mode)")
    parser.add_argument("--nlp-model", default=DEFAULT_NLP_MODEL)
    parser.add_argument("--max-chunk-seconds", type=float, default=60)

    args = parser.parse_args()

    try:
        if args.from_artifact:
            asr_artifact_path = args.from_artifact
            if not asr_artifact_path.exists():
                logger.error(f"ASR artifact not found: {asr_artifact_path}")
                sys.exit(1)
        else:
            asr_artifact_path = run_asr_first(args.input, args.asr_model, args.pre)

        policy = ChunkingPolicy(max_chunk_seconds=args.max_chunk_seconds)

        result, artifact_path = run_ner(
            asr_artifact_path,
            nlp_model=args.nlp_model,
            chunking_policy=policy,
        )

        # Print entities by type
        entities = result["output"]["entities"]
        type_counts = result["output"]["entity_type_counts"]

        print(f"\n--- Named Entities ({len(entities)}) ---")
        for entity_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"\n  {entity_type} ({count}):")
            type_entities = [e for e in entities if e["type"] == entity_type]
            for e in type_entities[:5]:  # Show top 5 per type
                print(f"    - {e['text']} (×{e['count']})")
            if len(type_entities) > 5:
                print(f"    ... and {len(type_entities) - 5} more")

        print(f"\n🎉 NER completed! Artifact: {artifact_path}")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
