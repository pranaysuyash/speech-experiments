#!/usr/bin/env python3
"""
Action Items From ASR - Extract structured action items from transcript.

Supports multi-chunk extraction for long audio:
1. Chunk transcript using chunking module
2. Extract action items from each chunk
3. Merge and dedupe across chunks
"""

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.nlp_schema import (
    NLPArtifact, NLPRunContext, NLPInputs, NLPProvenance,
    ActionItem, ActionItemsOutput,
    compute_text_hash, compute_file_hash, 
    load_asr_artifact, validate_nlp_artifact,
    ACTION_ITEMS_PROMPT_TEMPLATE,
)
from harness.llm_provider import get_llm_completion, LLMResult, ErrorCode
from harness.transcript_view import from_asr_artifact, TranscriptView
from harness.chunking import chunk_transcript, ChunkingPolicy, ChunkingResult, Chunk, dedupe_items

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_action_items")

DEFAULT_NLP_MODEL = "gemini-2.0-flash"
MAX_FINAL_ITEMS = 20  # Operability limit
DEDUPE_THRESHOLD = 0.7  # Jaccard similarity threshold


def parse_action_items_response(text: str) -> List[Dict]:
    """Parse JSON response into list of action item dicts."""
    text = text.strip()
    
    json_match = re.search(r'\{[\s\S]*\}', text)
    if not json_match:
        logger.warning("No JSON found in response")
        return []
    
    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}")
        return []
    
    items = data.get('action_items', [])
    if not isinstance(items, list):
        return []
    
    result = []
    for item_dict in items:
        if not isinstance(item_dict, dict):
            continue
        text_val = item_dict.get('text', '')
        if text_val and isinstance(text_val, str) and len(text_val.strip()) >= 5:
            result.append({
                'text': text_val.strip(),
                'assignee': item_dict.get('assignee'),
                'due': item_dict.get('due'),
                'priority': item_dict.get('priority'),
            })
    
    return result


def extract_chunk_action_items(
    chunk: Chunk,
    nlp_model: str,
    asr_artifact_path: str,
    asr_artifact_hash: str,
    asr_model_id: str,
) -> Tuple[Dict[str, Any], Path, bool, List[Dict]]:
    """
    Extract action items from a single chunk.
    
    Returns:
        (artifact_dict, artifact_path, success, items)
    """
    prompt_hash = compute_text_hash(ACTION_ITEMS_PROMPT_TEMPLATE)
    
    prompt = ACTION_ITEMS_PROMPT_TEMPLATE.format(
        transcript=chunk.text[:15000]
    )
    
    result = get_llm_completion(
        prompt=prompt,
        model=nlp_model,
        text_hash=chunk.text_hash,
        prompt_hash=prompt_hash,
        max_attempts=3,
        use_cache=True,
    )
    
    if result.success:
        items = parse_action_items_response(result.text)
    else:
        items = []
    
    # Create ActionItem objects for output
    action_items = [ActionItem(**item) for item in items]
    
    output = ActionItemsOutput(
        action_items=action_items,
        total_items=len(action_items),
        source_word_count=chunk.word_count,
    )
    
    artifact = NLPArtifact(
        run_context=NLPRunContext(
            task="action_items_chunk",
            nlp_model_id=nlp_model,
            timestamp=datetime.now().isoformat(),
        ),
        inputs=NLPInputs(
            parent_artifact_path=asr_artifact_path,
            parent_artifact_hash=asr_artifact_hash,
            asr_model_id=asr_model_id,
            asr_text_hash=chunk.text_hash,
            transcript_word_count=chunk.word_count,
            audio_duration_s=chunk.end_s - chunk.start_s,
        ),
        provenance=NLPProvenance(
            prompt_template=ACTION_ITEMS_PROMPT_TEMPLATE[:100] + "...",
            prompt_hash=prompt_hash,
        ),
        output=output.to_dict(),
        metrics_structural={
            'latency_ms': result.latency_ms,
            'cached': result.cached,
            'chunk_id': chunk.chunk_id,
            'item_count': len(items),
        },
        gates={'has_failure': not result.success},
        errors=[f"{result.error_code}: {result.error_message}"] if not result.success else [],
    )
    
    # Save chunk artifact
    chunks_dir = Path('runs/nlp/action_items/chunks')
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    chunk_file = chunks_dir / f'chunk_{chunk.chunk_id}_{int(time.time())}.json'
    
    artifact_dict = artifact.to_dict()
    with open(chunk_file, 'w') as f:
        json.dump(artifact_dict, f, indent=2, default=str)
    
    return artifact_dict, chunk_file, result.success, items


def run_action_items(
    asr_artifact_path: Path,
    nlp_model: str = DEFAULT_NLP_MODEL,
    chunking_policy: Optional[ChunkingPolicy] = None,
) -> Tuple[Dict[str, Any], Path]:
    """
    Extract action items from ASR artifact, with chunking for long audio.
    """
    logger.info(f"=== Action Items from ASR: {asr_artifact_path.name} ===")
    
    # Load ASR artifact
    asr_artifact = load_asr_artifact(asr_artifact_path)
    asr_artifact_hash = compute_file_hash(asr_artifact_path)
    asr_model_id = asr_artifact['run_context']['model_id']
    audio_duration = asr_artifact['inputs'].get('audio_duration_s', 0)
    
    # Create transcript view
    view = from_asr_artifact(asr_artifact_path)
    logger.info(f"Transcript: {view.word_count} words, {view.duration_s:.1f}s")
    
    # Chunk transcript
    if chunking_policy is None:
        chunking_policy = ChunkingPolicy()
    
    chunk_result = chunk_transcript(view, chunking_policy)
    logger.info(f"Chunks: {chunk_result.total_chunks}")
    
    # Extract from each chunk
    all_items = []
    children = []
    
    for chunk in chunk_result.chunks:
        logger.info(f"  Chunk {chunk.chunk_id}: {chunk.word_count} words")
        
        chunk_artifact, chunk_path, success, items = extract_chunk_action_items(
            chunk=chunk,
            nlp_model=nlp_model,
            asr_artifact_path=str(asr_artifact_path),
            asr_artifact_hash=asr_artifact_hash,
            asr_model_id=asr_model_id,
        )
        
        children.append({
            'task': 'action_items_chunk',
            'path': str(chunk_path),
            'hash': compute_file_hash(chunk_path),
            'chunk_id': chunk.chunk_id,
            'text_hash': chunk.text_hash,
            'success': success,
            'item_count': len(items),
        })
        
        if success:
            all_items.extend(items)
    
    # Dedupe items
    items_before = len(all_items)
    deduped_items = dedupe_items(all_items, text_key='text', threshold=DEDUPE_THRESHOLD)
    items_after = len(deduped_items)
    
    logger.info(f"Dedupe: {items_before} → {items_after} items")
    
    # Enforce operability limit
    truncated = False
    if len(deduped_items) > MAX_FINAL_ITEMS:
        logger.warning(f"Truncating from {len(deduped_items)} to {MAX_FINAL_ITEMS}")
        deduped_items = deduped_items[:MAX_FINAL_ITEMS]
        truncated = True
    
    # Build final artifact
    action_items = [ActionItem(**item) for item in deduped_items]
    
    output = ActionItemsOutput(
        action_items=action_items,
        total_items=len(action_items),
        source_word_count=view.word_count,
    )
    
    violations = output.validate()
    
    prompt_hash = compute_text_hash(ACTION_ITEMS_PROMPT_TEMPLATE)
    
    artifact = NLPArtifact(
        run_context=NLPRunContext(
            task="action_items",
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
            prompt_template=ACTION_ITEMS_PROMPT_TEMPLATE[:100] + "...",
            prompt_hash=prompt_hash,
        ),
        output=output.to_dict(),
        metrics_structural={
            'chunk_count': chunk_result.total_chunks,
            'items_before_dedupe': items_before,
            'items_after_dedupe': items_after,
            'truncated': truncated,
        },
        gates={
            'has_failure': False,
            'has_operability_violations': len(violations) > 0,
        },
        errors=violations,
    )
    
    validate_nlp_artifact(artifact)
    
    # Add chunking and merge info
    artifact_dict = artifact.to_dict()
    artifact_dict['chunking'] = chunk_result.policy.to_dict()
    artifact_dict['children'] = children
    artifact_dict['merge'] = {
        'dedupe_policy': 'jaccard',
        'similarity_threshold': DEDUPE_THRESHOLD,
        'items_before': items_before,
        'items_after': items_after,
        'truncated': truncated,
        'max_items': MAX_FINAL_ITEMS,
    }
    
    # Save
    runs_dir = Path('runs/nlp/action_items')
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_file = runs_dir / f'action_items_{int(time.time())}.json'
    
    with open(run_file, 'w') as f:
        json.dump(artifact_dict, f, indent=2, default=str)
    
    logger.info(f"✓ Action items artifact saved: {run_file}")
    print(f"ARTIFACT_PATH:{run_file}")
    
    return artifact_dict, run_file


def run_asr_first(input_path: Path, asr_model: Optional[str] = None, pre: Optional[str] = None) -> Path:
    """Run ASR on input file and return artifact path."""
    logger.info(f"Running ASR on {input_path.name}...")
    
    if asr_model:
        cmd = [sys.executable, str(Path(__file__).parent / "run_asr.py"),
               "--model", asr_model, "--input", str(input_path.resolve())]
    else:
        cmd = [sys.executable, str(Path(__file__).parent / "model_app.py"),
               "recommend", "--task", "asr", "--audio", str(input_path.resolve())]
    
    if pre:
        cmd.extend(["--pre", pre])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"ASR failed: {result.stderr[-200:]}")
    
    for line in result.stdout.split('\n'):
        if line.startswith("ARTIFACT_PATH:"):
            return Path(line.split(":", 1)[1].strip())
    
    raise RuntimeError("ASR did not produce artifact path")


def main():
    parser = argparse.ArgumentParser(description="Extract action items from ASR transcript")
    
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
        
        result, artifact_path = run_action_items(
            asr_artifact_path,
            nlp_model=args.nlp_model,
            chunking_policy=policy,
        )
        
        # Print action items
        items = result['output']['action_items']
        print(f"\n--- Action Items ({len(items)}) ---")
        for i, item in enumerate(items, 1):
            assignee = f" [{item.get('assignee', 'unassigned')}]" if item.get('assignee') else ""
            due = f" (due: {item['due']})" if item.get('due') else ""
            print(f"  {i}. {item['text']}{assignee}{due}")
        
        chunk_count = result.get('metrics_structural', {}).get('chunk_count', 1)
        if chunk_count > 1:
            print(f"\n📄 Artifact: {artifact_path}")
            print(f"   ({chunk_count} chunks, dedupe applied)")
        else:
            print(f"\n🎉 Action items extracted! Artifact: {artifact_path}")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
