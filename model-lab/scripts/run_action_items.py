#!/usr/bin/env python3
"""
Action Items From ASR - Extract structured action items from transcript.

This is an audio-derived NLP task. It consumes an ASR artifact and produces
structured action items with full provenance linkage.

Two modes:
1. Pipeline mode: --from-artifact <asr_artifact.json>
2. Convenience mode: --input <audio/video> (auto-runs ASR first)

Example usage:
    # Pipeline mode
    python scripts/run_action_items.py --from-artifact runs/.../adhoc_123.json
    
    # Convenience mode  
    python scripts/run_action_items.py --input meeting.mp4 --pre trim_silence
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

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.nlp_schema import (
    NLPArtifact, NLPRunContext, NLPInputs, NLPProvenance,
    ActionItem, ActionItemsOutput,
    compute_text_hash, compute_file_hash, 
    load_asr_artifact, validate_nlp_artifact,
    ACTION_ITEMS_PROMPT_TEMPLATE,
)
from harness.llm_provider import get_llm_completion, LLMResult, ErrorCode

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_action_items")

DEFAULT_NLP_MODEL = "gemini-2.0-flash"


def parse_action_items_response(text: str) -> List[ActionItem]:
    """Parse JSON response into list of ActionItem objects."""
    # Try to extract JSON from response
    text = text.strip()
    
    # Find JSON block
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
        if not text_val or not isinstance(text_val, str):
            continue
        
        result.append(ActionItem(
            text=text_val.strip(),
            assignee=item_dict.get('assignee'),
            due=item_dict.get('due'),
            priority=item_dict.get('priority'),
            evidence=item_dict.get('evidence', []),
        ))
    
    return result


def get_action_items(
    transcript: str, 
    transcript_hash: str,
    model: str = DEFAULT_NLP_MODEL,
    max_attempts: int = 3,
) -> Tuple[LLMResult, List[ActionItem]]:
    """
    Extract action items using LLM provider with retry and caching.
    
    Returns:
        (LLMResult, action_items) - Result includes success/failure status
    """
    prompt = ACTION_ITEMS_PROMPT_TEMPLATE.format(
        transcript=transcript[:15000]
    )
    
    prompt_hash = compute_text_hash(ACTION_ITEMS_PROMPT_TEMPLATE)
    
    result = get_llm_completion(
        prompt=prompt,
        model=model,
        text_hash=transcript_hash,
        prompt_hash=prompt_hash,
        max_attempts=max_attempts,
        use_cache=True,
    )
    
    if result.success:
        items = parse_action_items_response(result.text)
        return result, items
    else:
        return result, []


def run_asr_first(input_path: Path, asr_model: str = None, pre: str = None) -> Path:
    """Run ASR on input file and return artifact path."""
    logger.info(f"Running ASR on {input_path.name}...")
    
    if asr_model:
        cmd = [
            sys.executable, 
            str(Path(__file__).parent / "run_asr.py"),
            "--model", asr_model,
            "--input", str(input_path.resolve())
        ]
    else:
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
    
    artifact_path = None
    for line in result.stdout.split('\n'):
        if line.startswith("ARTIFACT_PATH:"):
            artifact_path = line.split(":", 1)[1].strip()
            break
        if "Artifact:" in line:
            artifact_path = line.split("Artifact:", 1)[1].strip()
            break
    
    if not artifact_path:
        raise RuntimeError("ASR did not produce artifact path")
    
    return Path(artifact_path)


def run_action_items(
    asr_artifact_path: Path,
    nlp_model: str = DEFAULT_NLP_MODEL,
) -> Tuple[Dict[str, Any], Path]:
    """
    Extract action items from ASR artifact.
    
    Returns:
        (artifact_dict, artifact_path)
    """
    logger.info(f"=== Action Items from ASR: {asr_artifact_path.name} ===")
    
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
    
    # Compute prompt hash
    prompt_hash = compute_text_hash(ACTION_ITEMS_PROMPT_TEMPLATE)
    
    # Extract action items
    logger.info("Extracting action items...")
    llm_result, items = get_action_items(
        transcript, 
        transcript_hash, 
        model=nlp_model,
        max_attempts=3,
    )
    
    # Build output with validation
    is_failure = not llm_result.success
    
    action_items_output = ActionItemsOutput(
        action_items=items,
        total_items=len(items),
        source_word_count=word_count,
    )
    
    # Validate operability thresholds
    violations = action_items_output.validate()
    if violations:
        for v in violations:
            logger.warning(f"Operability violation: {v}")
    
    if llm_result.success:
        logger.info(f"Extracted {len(items)} action items")
        if llm_result.cached:
            logger.info("(from cache)")
    else:
        logger.warning(f"LLM failed: {llm_result.error_code} - {llm_result.error_message}")
    
    # Build artifact
    run_context = NLPRunContext(
        task="action_items",
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
        prompt_template=ACTION_ITEMS_PROMPT_TEMPLATE[:100] + "...",
        prompt_hash=prompt_hash,
        has_ground_truth=False,
        metrics_valid=True,
    )
    
    structural_metrics = {
        'latency_ms': llm_result.latency_ms,
        'item_count': len(items),
        'cached': llm_result.cached,
        'attempts': llm_result.attempts,
        'operability_violations': len(violations),
    }
    
    errors = violations.copy()
    if is_failure:
        errors.append(f"{llm_result.error_code}: {llm_result.error_message}")
    
    artifact = NLPArtifact(
        run_context=run_context,
        inputs=inputs,
        provenance=provenance,
        output=action_items_output.to_dict(),
        metrics_structural=structural_metrics,
        gates={
            'has_failure': is_failure,
            'error_code': llm_result.error_code if is_failure else None,
            'has_operability_violations': len(violations) > 0,
        },
        errors=errors,
    )
    
    # Validate
    validate_nlp_artifact(artifact)
    
    # Save artifact
    runs_dir = Path('runs/nlp/action_items')
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    run_file = runs_dir / f'action_items_{timestamp}.json'
    
    result_dict = artifact.to_dict()
    with open(run_file, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)
    
    logger.info(f"✓ Action items artifact saved: {run_file}")
    print(f"ARTIFACT_PATH:{run_file}")
    
    return result_dict, run_file


def main():
    parser = argparse.ArgumentParser(description="Extract action items from ASR transcript")
    
    # Input modes (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--from-artifact", type=Path, 
                            help="ASR artifact path (pipeline mode)")
    input_group.add_argument("--input", type=Path,
                            help="Audio/video file (convenience mode, runs ASR first)")
    
    # Options
    parser.add_argument("--asr-model", help="ASR model to use (convenience mode only)")
    parser.add_argument("--pre", help="Preprocessing operators for ASR (convenience mode)")
    parser.add_argument("--nlp-model", default=DEFAULT_NLP_MODEL,
                       help=f"NLP model (default: {DEFAULT_NLP_MODEL})")
    
    args = parser.parse_args()
    
    try:
        # Determine ASR artifact
        if args.from_artifact:
            asr_artifact_path = args.from_artifact
            if not asr_artifact_path.exists():
                logger.error(f"ASR artifact not found: {asr_artifact_path}")
                sys.exit(1)
        else:
            if not args.input.exists():
                logger.error(f"Input file not found: {args.input}")
                sys.exit(1)
            
            asr_artifact_path = run_asr_first(
                args.input, 
                asr_model=args.asr_model,
                pre=args.pre
            )
        
        # Extract action items
        result, artifact_path = run_action_items(
            asr_artifact_path,
            nlp_model=args.nlp_model,
        )
        
        # Print action items
        items = result['output']['action_items']
        print(f"\n--- Action Items ({len(items)}) ---")
        for i, item in enumerate(items, 1):
            assignee = f" [{item.get('assignee', 'unassigned')}]" if item.get('assignee') else ""
            due = f" (due: {item['due']})" if item.get('due') else ""
            print(f"  {i}. {item['text']}{assignee}{due}")
        
        print(f"\n🎉 Action items extracted! Artifact: {artifact_path}")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
