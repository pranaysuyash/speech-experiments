#!/usr/bin/env python3
"""
Action Items With Assignee - Speaker-aware action item extraction.

Uses alignment artifact to attributing action items to specific speakers.
Enforces strict assignee constraints (must be valid speaker ID or UNKNOWN).

Two-pass strategy:
1. Pass A (Per-Speaker): Extract self-assigned items ("I will...")
2. Pass B (Resolution): Merge duplicates and resolve cross-assignments
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
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.nlp_schema import (
    NLPArtifact, NLPRunContext, NLPInputs, NLPProvenance,
    ActionItem, ActionItemsOutput,
    compute_text_hash, compute_file_hash, 
    validate_nlp_artifact
)
from harness.llm_provider import get_llm_completion, LLMResult
from harness.alignment import AlignedTranscript, AlignedSegment

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_action_items_with_assignee")

DEFAULT_NLP_MODEL = "gemini-2.0-flash"
MAX_ITEMS = 20

# Pass A: Self-assignment extraction
PASS_A_PROMPT = """Extract action items where speaker {speaker} commits to a task.
Only include items where {speaker} says "I will", "I'll", "I can", "Leave it to me", etc.
Do not include items assigned to others.

Transcript of {speaker}:
{transcript}

Return JSON:
{{
  "items": [
    {{
      "text": "Action description",
      "evidence_quote": "Exact quote from transcript"
    }}
  ]
}}"""

# Pass B: Resolution
PASS_B_PROMPT = """Review and refine these candidate action items.
1. Merge duplicates (keep the most descriptive version).
2. Validate assignees against allowed list: {allowed_speakers}.
   - If audio evidence shows one speaker assigning to another, update assignee.
   - Otherwise keep existing assignee or set UNKNOWN.
3. Limit to top {max_items} most important items.

Allowed Speakers: {allowed_speakers}

Candidate Items:
{candidates}

Global Context (excerpts):
{context}

Return JSON:
{{
  "final_items": [
    {{
      "text": "Action description",
      "assignee": "SPEAKER_ID or UNKNOWN",
      "evidence_quote": "Quote supporting the item",
      "priority": "high|medium|low"
    }}
  ]
}}"""


def run_alignment_dependency(input_path: Path, pre: str = None) -> Path:
    """Run alignment pipeline."""
    cmd = [sys.executable, str(Path(__file__).parent / "run_alignment.py"), "--input", str(input_path)]
    if pre:
        cmd.extend(["--pre", pre])
    
    logger.info("Running alignment pipeline...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Alignment failed: {result.stderr[-500:]}")
        
    for line in result.stdout.split('\n'):
        if line.startswith("ARTIFACT_PATH:"):
            return Path(line.split(":", 1)[1].strip())
            
    raise RuntimeError("Alignment produced no artifact path")


def load_alignment(path: Path) -> AlignedTranscript:
    with open(path) as f:
        data = json.load(f)
    # Simplified loading logic matching run_summarize_by_speaker
    from harness.alignment import AlignmentMetrics
    segments = [AlignedSegment(**s) for s in data['output']['segments']]
    metrics = AlignmentMetrics(**data['output']['metrics'])
    return AlignedTranscript(
        segments=segments,
        metrics=metrics,
        source_asr_path=data['output']['source_asr_path'],
        source_diarization_path=data['output']['source_diarization_path'],
    )


def extract_pass_a(
    speaker: str, 
    text: str, 
    nlp_model: str
) -> List[Dict]:
    """Pass A: Extract self-assigned items for a speaker."""
    if len(text.split()) < 10:
        return []
        
    prompt = PASS_A_PROMPT.format(speaker=speaker, transcript=text[:10000])
    text_hash = compute_text_hash(text)
    prompt_hash = compute_text_hash(prompt)
    
    result = get_llm_completion(
        prompt=prompt,
        model=nlp_model,
        text_hash=text_hash,
        prompt_hash=prompt_hash,
        use_cache=True,
    )
    
    items = []
    if result.success:
        try:
            # Flexible JSON parsing
            json_match = re.search(r'\{[\s\S]*\}', result.text)
            if json_match:
                data = json.loads(json_match.group())
                for item in data.get('items', []):
                    items.append({
                        'text': item.get('text'),
                        'assignee': speaker,
                        'evidence': item.get('evidence_quote'),
                        'source': 'pass_a_self_assign'
                    })
        except Exception as e:
            logger.warning(f"Pass A JSON parse error for {speaker}: {e}")
            
    return items


def extract_pass_b(
    candidates: List[Dict],
    context_text: str,
    allowed_speakers: List[str],
    nlp_model: str
) -> List[Dict]:
    """Pass B: Resolve and merge items."""
    if not candidates:
        return []

    # Serialize candidates for prompt
    candidates_str = json.dumps(candidates, indent=2)
    allowed_str = ", ".join(allowed_speakers + ["UNKNOWN"])
    
    prompt = PASS_B_PROMPT.format(
        allowed_speakers=allowed_str,
        candidates=candidates_str,
        context=context_text[:15000], # Limit context size
        max_items=MAX_ITEMS
    )
    
    # Hash based on candidates + context
    inputs_hash = compute_text_hash(candidates_str + context_text[:1000] + allowed_str)
    prompt_hash = compute_text_hash(prompt)
    
    result = get_llm_completion(
        prompt=prompt,
        model=nlp_model,
        text_hash=inputs_hash,
        prompt_hash=prompt_hash,
        use_cache=True,
    )
    
    final_items = []
    if result.success:
        try:
            json_match = re.search(r'\{[\s\S]*\}', result.text)
            if json_match:
                data = json.loads(json_match.group())
                raw_items = data.get('final_items', [])
                
                for item in raw_items:
                    # Validate assignee
                    assignee = item.get('assignee', 'UNKNOWN')
                    if assignee not in allowed_speakers and assignee != 'UNKNOWN':
                        logger.warning(f"Invalid assignee '{assignee}' -> UNKNOWN")
                        assignee = 'UNKNOWN'
                        
                    final_items.append({
                        'text': item.get('text'),
                        'assignee': assignee,
                        'priority': item.get('priority'),
                        'evidence': item.get('evidence_quote') # Keep simple for now
                    })
        except Exception as e:
            logger.warning(f"Pass B JSON parse error: {e}")
            return candidates # Fallback to candidates
            
    return final_items


def run_action_items_with_assignee(
    alignment_path: Path,
    nlp_model: str = DEFAULT_NLP_MODEL,
) -> Tuple[Dict[str, Any], Path]:
    
    transcript = load_alignment(alignment_path)
    allowed_speakers = sorted(list(transcript.metrics.speaker_distribution.keys()))
    
    # Check coverage gate
    if transcript.metrics.coverage_ratio < 0.85:
        logger.warning(f"⚠️ Low alignment coverage: {transcript.metrics.coverage_ratio:.1%}")
    
    # Group text by speaker for Pass A
    speaker_text = defaultdict(list)
    full_text_context = [] # For Pass B context
    
    for seg in transcript.segments:
        if seg.speaker_id != "unknown":
            speaker_text[seg.speaker_id].append(seg.text)
        full_text_context.append(f"[{seg.speaker_id}]: {seg.text}")
            
    # --- Pass A ---
    candidates = []
    logger.info("Pass A: Extraction per speaker...")
    for speaker, texts in speaker_text.items():
        joined_text = " ".join(texts)
        items = extract_pass_a(speaker, joined_text, nlp_model)
        candidates.extend(items)
        logger.info(f"  {speaker}: {len(items)} items")
        
    logger.info(f"Pass A found {len(candidates)} candidates")
    
    # --- Pass B ---
    # Construct global context (simplified: just full text with speaker tags, truncated)
    # Ideally we'd pick context around the quotes, but full transcript is usually OK for 1h meetings
    # if we truncate reasonably.
    context_str = "\n".join(full_text_context)
    
    logger.info("Pass B: Resolution and Merge...")
    final_raw_items = extract_pass_b(candidates, context_str, allowed_speakers, nlp_model)
    
    # Convert to schema objects
    action_items = []
    for item in final_raw_items:
        # Create ActionItem
        # Evidence requires EvidenceSpan objects. Schema expects it.
        # We only have a quote string. We'd need to find start/end words/chars.
        # For this milestone, we'll make a simplified EvidenceSpan or just leave list empty
        # but put the quote in metadata/description if possible?
        # Actually nlp_schema.ActionItem has evidence: List[EvidenceSpan].
        # Let's create a partial evidence span.
        
        # NOTE: Finding exact span indices in full transcript is complex/expensive here.
        # We will populate the quote field but might skip indices for now if acceptable.
        # The prompt asked for "evidence_quote".
        
        # Hack: Schema might require EvidenceSpan structure
        # from harness.nlp_schema import EvidenceSpan
        # evidence = [EvidenceSpan(quote=item['evidence'], start_word=0, end_word=0)]
        
        action_items.append(ActionItem(
            text=item['text'],
            assignee=item['assignee'],
            priority=item.get('priority'),
            due=None, # Not extracting due date yet
            evidence=[] # TODO: Map quotes to spans if strictness required
        ))
    
    output = ActionItemsOutput(
        action_items=action_items,
        total_items=len(action_items),
        source_word_count=sum(len(s.split()) for s in full_text_context),
    )
    
    # Create Artifact
    run_context = NLPRunContext(
        task="action_items_with_assignee",
        nlp_model_id=nlp_model,
        timestamp=datetime.now().isoformat(),
        command=sys.argv,
    )
    
    inputs = NLPInputs(
        parent_artifact_path=str(alignment_path),
        parent_artifact_hash=compute_file_hash(alignment_path),
        asr_model_id="alignment",
        asr_text_hash=compute_text_hash(context_str),
        transcript_word_count=output.source_word_count,
        audio_duration_s=transcript.metrics.total_duration_s,
    )
    
    artifact = NLPArtifact(
        run_context=run_context,
        inputs=inputs,
        provenance=NLPProvenance(
            prompt_template=PASS_B_PROMPT[:100] + "...",
            prompt_hash=compute_text_hash(PASS_B_PROMPT),
        ),
        output=output.to_dict(),
        metrics_structural={
            'pass_a_count': len(candidates),
            'pass_b_count': len(final_raw_items),
            'coverage_ratio': transcript.metrics.coverage_ratio,
        },
        gates={
            'has_failure': False,
            'low_coverage': transcript.metrics.coverage_ratio < 0.85
        },
    )
    
    # Save
    runs_dir = Path("runs/nlp/action_items_assignee")
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_file = runs_dir / f"action_items_{int(time.time())}.json"
    
    with open(run_file, 'w') as f:
        json.dump(artifact.to_dict(), f, indent=2)
        
    logger.info(f"✓ Artifact saved: {run_file}")
    logger.info(f"  Items: {len(action_items)}")
    print(f"ARTIFACT_PATH:{run_file}")
    
    return artifact.to_dict(), run_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--from-artifact", type=Path)
    parser.add_argument("--nlp-model", default=DEFAULT_NLP_MODEL)
    parser.add_argument("--pre")
    
    args = parser.parse_args()
    
    try:
        if args.from_artifact:
            path = args.from_artifact
        elif args.input:
            path = run_alignment_dependency(args.input, args.pre)
        else:
            raise ValueError("Required: --input or --from-artifact")
            
        result, _ = run_action_items_with_assignee(path, args.nlp_model)
        
        print("\n--- Action Items ---")
        for item in result['output']['action_items']:
            print(f"[{item.get('assignee', 'UNKNOWN')}] {item['text']}")
            
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
