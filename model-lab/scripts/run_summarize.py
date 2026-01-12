#!/usr/bin/env python3
"""
Summarize From ASR - Generate constrained summary from transcript.

This is an audio-derived NLP task. It consumes an ASR artifact and produces
a structured summary with full provenance linkage.

Two modes:
1. Pipeline mode: --from-artifact <asr_artifact.json>
2. Convenience mode: --input <audio/video> (auto-runs ASR first)

Example usage:
    # Pipeline mode
    python scripts/run_summarize.py --from-artifact runs/.../adhoc_123.json
    
    # Convenience mode  
    python scripts/run_summarize.py --input meeting.mp4 --pre trim_silence
    
    # With model selection
    python scripts/run_summarize.py --input meeting.mp4 --asr-model faster_whisper
"""

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.nlp_schema import (
    NLPArtifact, NLPRunContext, NLPInputs, NLPProvenance,
    SummaryOutput, compute_text_hash, compute_file_hash, 
    load_asr_artifact, validate_nlp_artifact,
    SUMMARY_PROMPT_TEMPLATE,
)

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_summarize")

# Default NLP model (can be overridden)
DEFAULT_NLP_MODEL = "gemini-2.0-flash"


def get_gemini_summary(transcript: str, max_sentences: int = 5) -> Tuple[List[str], Dict[str, Any]]:
    """
    Generate summary using Gemini API.
    
    Returns:
        (sentences, metrics) - List of summary sentences and API metrics
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    
    genai.configure(api_key=api_key)
    
    # Build prompt
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        max_sentences=max_sentences,
        transcript=transcript[:15000]  # Limit transcript length
    )
    
    # Call API
    t0 = time.time()
    model = genai.GenerativeModel(DEFAULT_NLP_MODEL)
    response = model.generate_content(prompt)
    latency_ms = (time.time() - t0) * 1000
    
    # Parse response into sentences
    raw_text = response.text.strip()
    
    # Extract bullet points or sentences
    lines = raw_text.split('\n')
    sentences = []
    for line in lines:
        line = line.strip()
        # Remove bullet markers
        line = re.sub(r'^[-•*]\s*', '', line)
        line = re.sub(r'^\d+\.\s*', '', line)
        if line and len(line) > 10:  # Skip very short lines
            sentences.append(line)
    
    # Limit to max_sentences
    sentences = sentences[:max_sentences]
    
    metrics = {
        'latency_ms': round(latency_ms, 1),
        'prompt_tokens': len(prompt.split()),  # Approximate
        'model': DEFAULT_NLP_MODEL,
    }
    
    return sentences, metrics


def run_asr_first(input_path: Path, asr_model: str = None, pre: str = None) -> Path:
    """
    Run ASR on input file and return artifact path.
    
    If asr_model not specified, uses decision engine to pick best.
    """
    logger.info(f"Running ASR on {input_path.name}...")
    
    # Build command
    if asr_model:
        # Direct runner call
        cmd = [
            sys.executable, 
            str(Path(__file__).parent / "run_asr.py"),
            "--model", asr_model,
            "--input", str(input_path.resolve())
        ]
    else:
        # Use decision engine
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "model_app.py"),
            "recommend", "--task", "asr",
            "--audio", str(input_path.resolve())
        ]
    
    if pre:
        cmd.extend(["--pre", pre])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"ASR failed: {result.stderr}")
        raise RuntimeError(f"ASR failed with exit code {result.returncode}")
    
    # Parse artifact path from output
    artifact_path = None
    for line in result.stdout.split('\n'):
        if line.startswith("ARTIFACT_PATH:"):
            artifact_path = line.split(":", 1)[1].strip()
            break
        # Also check for "Artifact:" from model_app
        if "Artifact:" in line:
            artifact_path = line.split("Artifact:", 1)[1].strip()
            break
    
    if not artifact_path:
        logger.error(f"Could not find artifact path in output:\n{result.stdout}")
        raise RuntimeError("ASR did not produce artifact path")
    
    return Path(artifact_path)


def run_summarize(
    asr_artifact_path: Path,
    max_sentences: int = 5,
    nlp_model: str = DEFAULT_NLP_MODEL,
) -> Tuple[Dict[str, Any], Path]:
    """
    Generate summary from ASR artifact.
    
    Args:
        asr_artifact_path: Path to ASR artifact JSON
        max_sentences: Maximum sentences in summary
        nlp_model: NLP model to use
        
    Returns:
        (artifact_dict, artifact_path)
    """
    logger.info(f"=== Summarize from ASR: {asr_artifact_path.name} ===")
    
    # Load and validate ASR artifact
    asr_artifact = load_asr_artifact(asr_artifact_path)
    asr_artifact_hash = compute_file_hash(asr_artifact_path)
    
    # Extract transcript
    transcript = asr_artifact['output']['text']
    transcript_hash = compute_text_hash(transcript)
    word_count = len(transcript.split())
    
    logger.info(f"Transcript: {word_count} words, hash={transcript_hash[:12]}")
    
    # Get ASR metadata
    asr_model_id = asr_artifact['run_context']['model_id']
    audio_duration = asr_artifact['inputs'].get('audio_duration_s', 0)
    
    # Compute prompt hash for reproducibility
    prompt_hash = compute_text_hash(SUMMARY_PROMPT_TEMPLATE)
    
    # Generate summary
    logger.info(f"Generating summary (max {max_sentences} sentences)...")
    try:
        sentences, api_metrics = get_gemini_summary(transcript, max_sentences)
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        raise
    
    # Build output
    summary_word_count = sum(len(s.split()) for s in sentences)
    compression_ratio = word_count / summary_word_count if summary_word_count > 0 else 0
    
    summary_output = SummaryOutput(
        sentences=sentences,
        total_sentences=len(sentences),
        source_word_count=word_count,
        compression_ratio=compression_ratio,
    )
    
    logger.info(f"Summary: {len(sentences)} sentences, compression={compression_ratio:.1f}x")
    
    # Build artifact
    run_context = NLPRunContext(
        task="summarize",
        nlp_model_id=nlp_model,
        timestamp=datetime.now().isoformat(),
        git_hash=None,
        command=sys.argv,
    )
    
    inputs = NLPInputs(
        parent_artifact_path=str(asr_artifact_path),
        parent_artifact_hash=asr_artifact_hash,
        asr_model_id=asr_model_id,
        asr_text_hash=transcript_hash,
        transcript_word_count=word_count,
        audio_duration_s=audio_duration,
    )
    
    provenance = NLPProvenance(
        prompt_template=SUMMARY_PROMPT_TEMPLATE[:100] + "...",  # Truncate for storage
        prompt_hash=prompt_hash,
        has_ground_truth=False,
        metrics_valid=True,
    )
    
    structural_metrics = {
        'latency_ms': api_metrics.get('latency_ms'),
        'summary_sentences': len(sentences),
        'summary_word_count': summary_word_count,
        'compression_ratio': round(compression_ratio, 2),
        'max_sentences_requested': max_sentences,
    }
    
    artifact = NLPArtifact(
        run_context=run_context,
        inputs=inputs,
        provenance=provenance,
        output=summary_output.to_dict(),
        metrics_structural=structural_metrics,
        gates={'has_failure': len(sentences) == 0},
        errors=[],
    )
    
    # Validate
    validate_nlp_artifact(artifact)
    
    # Save artifact
    runs_dir = Path(f'runs/nlp/summarize')
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    run_file = runs_dir / f'summary_{timestamp}.json'
    
    result_dict = artifact.to_dict()
    with open(run_file, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)
    
    logger.info(f"✓ Summary artifact saved: {run_file}")
    print(f"ARTIFACT_PATH:{run_file}")
    
    return result_dict, run_file


def main():
    parser = argparse.ArgumentParser(description="Generate summary from ASR transcript")
    
    # Input modes (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--from-artifact", type=Path, 
                            help="ASR artifact path (pipeline mode)")
    input_group.add_argument("--input", type=Path,
                            help="Audio/video file (convenience mode, runs ASR first)")
    
    # Options
    parser.add_argument("--max-sentences", type=int, default=5,
                       help="Maximum sentences in summary (default: 5)")
    parser.add_argument("--asr-model", help="ASR model to use (convenience mode only)")
    parser.add_argument("--pre", help="Preprocessing operators for ASR (convenience mode)")
    parser.add_argument("--nlp-model", default=DEFAULT_NLP_MODEL,
                       help=f"NLP model (default: {DEFAULT_NLP_MODEL})")
    
    args = parser.parse_args()
    
    try:
        # Determine ASR artifact
        if args.from_artifact:
            # Pipeline mode
            asr_artifact_path = args.from_artifact
            if not asr_artifact_path.exists():
                logger.error(f"ASR artifact not found: {asr_artifact_path}")
                sys.exit(1)
        else:
            # Convenience mode - run ASR first
            if not args.input.exists():
                logger.error(f"Input file not found: {args.input}")
                sys.exit(1)
            
            asr_artifact_path = run_asr_first(
                args.input, 
                asr_model=args.asr_model,
                pre=args.pre
            )
        
        # Generate summary
        result, artifact_path = run_summarize(
            asr_artifact_path,
            max_sentences=args.max_sentences,
            nlp_model=args.nlp_model,
        )
        
        # Print summary
        print("\n--- Summary ---")
        for i, s in enumerate(result['output']['sentences'], 1):
            print(f"  {i}. {s}")
        
        print(f"\n🎉 Summary completed! Artifact: {artifact_path}")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
