#!/usr/bin/env python3
"""
Summarize From ASR - Generate constrained summary from transcript.

Supports multi-chunk summarization for long audio:
1. Chunk transcript using chunking module
2. Summarize each chunk → chunk artifacts
3. Aggregate chunk summaries → final artifact

Two modes:
1. Pipeline mode: --from-artifact <asr_artifact.json>
2. Convenience mode: --input <audio/video> (auto-runs ASR first)
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
    SummaryOutput, compute_text_hash, compute_file_hash, 
    load_asr_artifact, validate_nlp_artifact,
    SUMMARY_PROMPT_TEMPLATE,
)
from harness.llm_provider import get_llm_completion, LLMResult, ErrorCode
from harness.transcript_view import from_asr_artifact, TranscriptView
from harness.chunking import chunk_transcript, ChunkingPolicy, ChunkingResult, Chunk

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_summarize")

DEFAULT_NLP_MODEL = "gemini-2.0-flash"
MIN_SUCCESS_RATIO = 0.8  # For partial failure handling

# Aggregate prompt for combining chunk summaries
AGGREGATE_PROMPT_TEMPLATE = """Combine the following section summaries into a unified summary.
Create {max_sentences} sentences or fewer that capture the key points across all sections.
Do not add information not present in the summaries.

Section Summaries:
{chunk_summaries}

Combined Summary (bullet points):"""


def parse_summary_response(text: str, max_sentences: int) -> List[str]:
    """Parse LLM response into list of sentences."""
    lines = text.split('\n')
    sentences = []
    for line in lines:
        line = line.strip()
        line = re.sub(r'^[-•*]\s*', '', line)
        line = re.sub(r'^\d+\.\s*', '', line)
        if line and len(line) > 10:
            sentences.append(line)
    return sentences[:max_sentences]


def summarize_chunk(
    chunk: Chunk,
    max_sentences: int,
    nlp_model: str,
    asr_artifact_path: str,
    asr_artifact_hash: str,
    asr_model_id: str,
) -> Tuple[Dict[str, Any], Path, bool]:
    """
    Summarize a single chunk and save artifact.
    
    Returns:
        (artifact_dict, artifact_path, success)
    """
    prompt_hash = compute_text_hash(SUMMARY_PROMPT_TEMPLATE)
    
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        max_sentences=max_sentences,
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
        sentences = parse_summary_response(result.text, max_sentences)
    else:
        sentences = []
    
    # Create chunk artifact
    summary_output = SummaryOutput(
        sentences=sentences,
        total_sentences=len(sentences),
        source_word_count=chunk.word_count,
        compression_ratio=chunk.word_count / sum(len(s.split()) for s in sentences) if sentences else 0,
    )
    
    run_context = NLPRunContext(
        task="summarize_chunk",
        nlp_model_id=nlp_model,
        timestamp=datetime.now().isoformat(),
    )
    
    inputs = NLPInputs(
        parent_artifact_path=asr_artifact_path,
        parent_artifact_hash=asr_artifact_hash,
        asr_model_id=asr_model_id,
        asr_text_hash=chunk.text_hash,
        transcript_word_count=chunk.word_count,
        audio_duration_s=chunk.end_s - chunk.start_s,
    )
    
    provenance = NLPProvenance(
        prompt_template=SUMMARY_PROMPT_TEMPLATE[:100] + "...",
        prompt_hash=prompt_hash,
    )
    
    artifact = NLPArtifact(
        run_context=run_context,
        inputs=inputs,
        provenance=provenance,
        output=summary_output.to_dict(),
        metrics_structural={
            'latency_ms': result.latency_ms,
            'cached': result.cached,
            'chunk_id': chunk.chunk_id,
            'chunk_start_s': chunk.start_s,
            'chunk_end_s': chunk.end_s,
        },
        gates={'has_failure': not result.success},
        errors=[f"{result.error_code}: {result.error_message}"] if not result.success else [],
    )
    
    # Save chunk artifact
    chunks_dir = Path('runs/nlp/summarize/chunks')
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    chunk_file = chunks_dir / f'chunk_{chunk.chunk_id}_{int(time.time())}.json'
    
    artifact_dict = artifact.to_dict()
    with open(chunk_file, 'w') as f:
        json.dump(artifact_dict, f, indent=2, default=str)
    
    return artifact_dict, chunk_file, result.success


def aggregate_summaries(
    chunk_results: List[Dict[str, Any]],
    max_sentences: int,
    nlp_model: str,
) -> Tuple[LLMResult, List[str]]:
    """
    Aggregate chunk summaries into final summary.
    """
    # Collect all chunk sentences
    all_sentences = []
    for chunk_result in chunk_results:
        sentences = chunk_result.get('output', {}).get('sentences', [])
        all_sentences.extend(sentences)
    
    if not all_sentences:
        return LLMResult(success=False, error_code=ErrorCode.INVALID_RESPONSE, error_message="No chunk summaries"), []
    
    # Build aggregate prompt
    chunk_summaries = "\n\n".join([
        f"Section {i+1}:\n" + "\n".join(f"- {s}" for s in chunk_result.get('output', {}).get('sentences', []))
        for i, chunk_result in enumerate(chunk_results)
    ])
    
    prompt = AGGREGATE_PROMPT_TEMPLATE.format(
        max_sentences=max_sentences,
        chunk_summaries=chunk_summaries
    )
    
    # Create unique hash for aggregate
    chunk_hashes = [r.get('inputs', {}).get('asr_text_hash', '') for r in chunk_results]
    aggregate_input_hash = compute_text_hash('|'.join(chunk_hashes))
    prompt_hash = compute_text_hash(AGGREGATE_PROMPT_TEMPLATE)
    
    result = get_llm_completion(
        prompt=prompt,
        model=nlp_model,
        text_hash=aggregate_input_hash,
        prompt_hash=prompt_hash,
        max_attempts=3,
        use_cache=True,
    )
    
    if result.success:
        sentences = parse_summary_response(result.text, max_sentences)
        return result, sentences
    else:
        return result, []


def run_summarize(
    asr_artifact_path: Path,
    max_sentences: int = 5,
    nlp_model: str = DEFAULT_NLP_MODEL,
    chunking_policy: ChunkingPolicy = None,
) -> Tuple[Dict[str, Any], Path]:
    """
    Generate summary from ASR artifact, with chunking for long audio.
    """
    logger.info(f"=== Summarize from ASR: {asr_artifact_path.name} ===")
    
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
    
    if chunk_result.error:
        logger.warning(f"Chunking warning: {chunk_result.error}")
    
    logger.info(f"Chunks: {chunk_result.total_chunks} (required={chunk_result.chunking_required})")
    
    # Single chunk path (original behavior)
    if not chunk_result.chunking_required:
        return _run_summarize_single(
            asr_artifact_path, asr_artifact, asr_artifact_hash,
            view, max_sentences, nlp_model
        )
    
    # Multi-chunk path
    return _run_summarize_chunked(
        asr_artifact_path, asr_artifact, asr_artifact_hash,
        view, chunk_result, max_sentences, nlp_model
    )


def _run_summarize_single(
    asr_artifact_path: Path,
    asr_artifact: Dict,
    asr_artifact_hash: str,
    view: TranscriptView,
    max_sentences: int,
    nlp_model: str,
) -> Tuple[Dict[str, Any], Path]:
    """Single-chunk summarization (original behavior)."""
    prompt_hash = compute_text_hash(SUMMARY_PROMPT_TEMPLATE)
    
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        max_sentences=max_sentences,
        transcript=view.full_text[:15000]
    )
    
    llm_result = get_llm_completion(
        prompt=prompt,
        model=nlp_model,
        text_hash=view.text_hash,
        prompt_hash=prompt_hash,
        max_attempts=3,
        use_cache=True,
    )
    
    if llm_result.success:
        sentences = parse_summary_response(llm_result.text, max_sentences)
        logger.info(f"Summary: {len(sentences)} sentences")
    else:
        sentences = []
        logger.warning(f"LLM failed: {llm_result.error_code}")
    
    # Build artifact
    summary_output = SummaryOutput(
        sentences=sentences,
        total_sentences=len(sentences),
        source_word_count=view.word_count,
        compression_ratio=view.word_count / sum(len(s.split()) for s in sentences) if sentences else 0,
    )
    
    asr_model_id = asr_artifact['run_context']['model_id']
    audio_duration = asr_artifact['inputs'].get('audio_duration_s', 0)
    
    artifact = NLPArtifact(
        run_context=NLPRunContext(
            task="summarize",
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
            prompt_template=SUMMARY_PROMPT_TEMPLATE[:100] + "...",
            prompt_hash=prompt_hash,
        ),
        output=summary_output.to_dict(),
        metrics_structural={
            'latency_ms': llm_result.latency_ms,
            'cached': llm_result.cached,
            'attempts': llm_result.attempts,
            'chunk_count': 1,
        },
        gates={'has_failure': not llm_result.success},
        errors=[f"{llm_result.error_code}: {llm_result.error_message}"] if not llm_result.success else [],
    )
    
    validate_nlp_artifact(artifact)
    
    # Save
    runs_dir = Path('runs/nlp/summarize')
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_file = runs_dir / f'summary_{int(time.time())}.json'
    
    result_dict = artifact.to_dict()
    with open(run_file, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)
    
    logger.info(f"✓ Summary artifact saved: {run_file}")
    print(f"ARTIFACT_PATH:{run_file}")
    
    return result_dict, run_file


def _run_summarize_chunked(
    asr_artifact_path: Path,
    asr_artifact: Dict,
    asr_artifact_hash: str,
    view: TranscriptView,
    chunk_result: ChunkingResult,
    max_sentences: int,
    nlp_model: str,
) -> Tuple[Dict[str, Any], Path]:
    """Multi-chunk summarization with aggregation."""
    asr_model_id = asr_artifact['run_context']['model_id']
    audio_duration = asr_artifact['inputs'].get('audio_duration_s', 0)
    
    # Summarize each chunk
    logger.info(f"Summarizing {chunk_result.total_chunks} chunks...")
    
    children = []
    successful_chunks = []
    failed_chunks = []
    
    for chunk in chunk_result.chunks:
        logger.info(f"  Chunk {chunk.chunk_id}: {chunk.word_count} words")
        
        chunk_artifact, chunk_path, success = summarize_chunk(
            chunk=chunk,
            max_sentences=max(3, max_sentences // chunk_result.total_chunks + 1),
            nlp_model=nlp_model,
            asr_artifact_path=str(asr_artifact_path),
            asr_artifact_hash=asr_artifact_hash,
            asr_model_id=asr_model_id,
        )
        
        children.append({
            'task': 'summarize_chunk',
            'path': str(chunk_path),
            'hash': compute_file_hash(chunk_path),
            'chunk_id': chunk.chunk_id,
            'text_hash': chunk.text_hash,
            'success': success,
        })
        
        if success:
            successful_chunks.append(chunk_artifact)
        else:
            failed_chunks.append(chunk.chunk_id)
    
    # Check success ratio
    success_ratio = len(successful_chunks) / len(chunk_result.chunks)
    
    if success_ratio < MIN_SUCCESS_RATIO:
        logger.error(f"Too many chunk failures: {len(failed_chunks)}/{len(chunk_result.chunks)}")
        status = "failed"
        sentences = []
        agg_result = LLMResult(success=False, error_code="CHUNK_FAILURE", error_message=f"Only {success_ratio:.0%} success")
    elif not successful_chunks:
        status = "failed"
        sentences = []
        agg_result = LLMResult(success=False, error_code="CHUNK_FAILURE", error_message="No successful chunks")
    else:
        # Aggregate successful chunk summaries
        logger.info(f"Aggregating {len(successful_chunks)} chunk summaries...")
        agg_result, sentences = aggregate_summaries(successful_chunks, max_sentences, nlp_model)
        
        if agg_result.success:
            status = "success" if not failed_chunks else "partial_success"
            logger.info(f"Aggregate: {len(sentences)} sentences")
        else:
            status = "failed"
            logger.warning(f"Aggregate failed: {agg_result.error_code}")
    
    # Build aggregate artifact
    summary_output = SummaryOutput(
        sentences=sentences,
        total_sentences=len(sentences),
        source_word_count=view.word_count,
        compression_ratio=view.word_count / sum(len(s.split()) for s in sentences) if sentences else 0,
    )
    
    prompt_hash = compute_text_hash(AGGREGATE_PROMPT_TEMPLATE)
    chunk_summary_hashes = [c.get('inputs', {}).get('asr_text_hash', '') for c in successful_chunks]
    
    artifact = NLPArtifact(
        run_context=NLPRunContext(
            task="summarize",
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
            prompt_template=AGGREGATE_PROMPT_TEMPLATE[:100] + "...",
            prompt_hash=prompt_hash,
        ),
        output=summary_output.to_dict(),
        metrics_structural={
            'latency_ms': agg_result.latency_ms if agg_result else 0,
            'cached': agg_result.cached if agg_result else False,
            'chunk_count': chunk_result.total_chunks,
            'successful_chunks': len(successful_chunks),
            'failed_chunks': failed_chunks,
        },
        gates={
            'has_failure': status == "failed",
            'status': status,
            'error_code': agg_result.error_code if not agg_result.success else None,
        },
        errors=[f"{agg_result.error_code}: {agg_result.error_message}"] if not agg_result.success else [],
    )
    
    # Add chunking info and children
    artifact_dict = artifact.to_dict()
    artifact_dict['chunking'] = chunk_result.policy.to_dict()
    artifact_dict['children'] = children
    artifact_dict['aggregation'] = {
        'model_id': nlp_model,
        'prompt_hash': prompt_hash,
        'input_chunk_summaries_hash': compute_text_hash('|'.join(chunk_summary_hashes)),
    }
    
    # Save aggregate artifact
    runs_dir = Path('runs/nlp/summarize')
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_file = runs_dir / f'summary_aggregate_{int(time.time())}.json'
    
    with open(run_file, 'w') as f:
        json.dump(artifact_dict, f, indent=2, default=str)
    
    logger.info(f"✓ Aggregate artifact saved: {run_file}")
    logger.info(f"  Children: {len(children)} chunk artifacts")
    print(f"ARTIFACT_PATH:{run_file}")
    
    return artifact_dict, run_file


def run_asr_first(input_path: Path, asr_model: str = None, pre: str = None) -> Path:
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
    parser = argparse.ArgumentParser(description="Generate summary from ASR transcript")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--from-artifact", type=Path, help="ASR artifact path")
    input_group.add_argument("--input", type=Path, help="Audio/video file")
    
    parser.add_argument("--max-sentences", type=int, default=5)
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
        
        result, artifact_path = run_summarize(
            asr_artifact_path,
            max_sentences=args.max_sentences,
            nlp_model=args.nlp_model,
            chunking_policy=policy,
        )
        
        # Print summary
        print("\n--- Summary ---")
        for i, s in enumerate(result['output']['sentences'], 1):
            print(f"  {i}. {s}")
        
        chunk_count = result.get('metrics_structural', {}).get('chunk_count', 1)
        if chunk_count > 1:
            print(f"\n📄 Aggregate artifact: {artifact_path}")
            print(f"   ({chunk_count} chunks processed)")
        else:
            print(f"\n🎉 Summary completed! Artifact: {artifact_path}")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
