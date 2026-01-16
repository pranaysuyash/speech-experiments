"""
NLP Tasks Runner - Logic for Summarization, Action Items, etc.
"""

import json
import logging
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from harness.nlp_schema import (
    NLPArtifact, NLPRunContext, NLPInputs, NLPProvenance,
    ActionItem, ActionItemsOutput,
    compute_text_hash, compute_file_hash, 
    validate_nlp_artifact
)
from harness.llm_provider import get_llm_completion, LLMResult
from harness.alignment import AlignedTranscript, AlignedSegment, load_alignment

logger = logging.getLogger("nlp")

DEFAULT_NLP_MODEL = "gemini-2.0-flash"

# --- Summarize By Speaker ---

SPEAKER_SUMMARY_PROMPT = """Summarize what {speaker} said/did in this meeting.
Focus on their key arguments, decisions, and action items.
Do not attribute other speakers' points to them.
Keep it concise (aim for < 5 bullets).

Transcript of {speaker}:
{transcript}

Summary (bullet points):"""

def summarize_speaker_content(
    speaker: str, 
    text: str, 
    nlp_model: str
) -> List[str]:
    """Summarize a single speaker's text."""
    if len(text.split()) < 20:
        return [] 
        
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
    
    if result.success:
        lines = result.text.strip().split('\n')
        bullets = [l.strip().lstrip('-•*').strip() for l in lines if len(l.strip()) > 5]
        return bullets
    return []

def run_summarize_by_speaker(
    alignment_path: Path,
    output_dir: Path,
    nlp_model: str = DEFAULT_NLP_MODEL,
    force: bool = False
) -> Path:
    """Run per-speaker summarization."""
    alignment_path = alignment_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading alignment: {alignment_path.name}")
    aligned_transcript = load_alignment(alignment_path)
    
    # Group by speaker
    speaker_text = defaultdict(list)
    for seg in aligned_transcript.segments:
        if seg.speaker_id != "unknown":
            speaker_text[seg.speaker_id].append(seg.text)
            
    speaker_summaries = {}
    for speaker, texts in speaker_text.items():
        full_text = " ".join(texts)
        bullets = summarize_speaker_content(speaker, full_text, nlp_model)
        if bullets:
            speaker_summaries[speaker] = bullets
            
    output = {
        'speaker_summaries': speaker_summaries,
        'speaker_stats': aligned_transcript.metrics.to_dict(),
    }
    
    run_context = NLPRunContext(
        task="summarize_by_speaker",
        nlp_model_id=nlp_model,
        timestamp=datetime.now().isoformat(),
        command=sys.argv, # Close enough
    )
    
    # Inputs provenance
    inputs = NLPInputs(
        parent_artifact_path=str(alignment_path),
        parent_artifact_hash=compute_file_hash(alignment_path),
        asr_model_id="alignment",
        asr_text_hash=compute_file_hash(alignment_path), 
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
        gates={'has_failure': False},
    )
    
    artifact_path = output_dir / f"summary_by_speaker_{alignment_path.stem.replace('alignment_', '')}.json"
    
    with open(artifact_path, 'w') as f:
        json.dump(artifact.to_dict(), f, indent=2)
        
    return artifact_path


# --- Action Items ---

MAX_ITEMS = 20
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

def extract_pass_a(speaker: str, text: str, nlp_model: str) -> List[Dict]:
    if len(text.split()) < 10: return []
    prompt = PASS_A_PROMPT.format(speaker=speaker, transcript=text[:10000])
    result = get_llm_completion(prompt, model=nlp_model, use_cache=True)
    items = []
    if result.success:
        try:
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

def extract_pass_b(candidates: List[Dict], context_text: str, allowed_speakers: List[str], nlp_model: str) -> List[Dict]:
    if not candidates: return []
    candidates_str = json.dumps(candidates, indent=2)
    allowed_str = ", ".join(allowed_speakers + ["UNKNOWN"])
    prompt = PASS_B_PROMPT.format(
        allowed_speakers=allowed_str,
        candidates=candidates_str,
        context=context_text[:15000],
        max_items=MAX_ITEMS
    )
    result = get_llm_completion(prompt, model=nlp_model, use_cache=True)
    final_items = []
    if result.success:
        try:
            json_match = re.search(r'\{[\s\S]*\}', result.text)
            if json_match:
                data = json.loads(json_match.group())
                raw_items = data.get('final_items', [])
                for item in raw_items:
                    assignee = item.get('assignee', 'UNKNOWN')
                    if assignee not in allowed_speakers and assignee != 'UNKNOWN':
                        assignee = 'UNKNOWN'
                    final_items.append({
                        'text': item.get('text'),
                        'assignee': assignee,
                        'priority': item.get('priority'),
                        'evidence': item.get('evidence_quote')
                    })
        except Exception:
            return candidates
    return final_items

def run_action_items(
    alignment_path: Path,
    output_dir: Path,
    nlp_model: str = DEFAULT_NLP_MODEL,
    force: bool = False
) -> Path:
    """Run action items extraction."""
    alignment_path = alignment_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    transcript = load_alignment(alignment_path)
    allowed_speakers = sorted(list(transcript.metrics.speaker_distribution.keys()))
    
    speaker_text = defaultdict(list)
    full_text_context = []
    for seg in transcript.segments:
        if seg.speaker_id != "unknown":
            speaker_text[seg.speaker_id].append(seg.text)
        full_text_context.append(f"[{seg.speaker_id}]: {seg.text}")
        
    candidates = []
    for speaker, texts in speaker_text.items():
        joined_text = " ".join(texts)
        items = extract_pass_a(speaker, joined_text, nlp_model)
        candidates.extend(items)
        
    context_str = "\n".join(full_text_context)
    final_raw_items = extract_pass_b(candidates, context_str, allowed_speakers, nlp_model)
    
    action_items = []
    for item in final_raw_items:
        action_items.append(ActionItem(
            text=item['text'],
            assignee=item['assignee'],
            priority=item.get('priority'),
            due=None,
            evidence=[] 
        ))
        
    output = ActionItemsOutput(
        action_items=action_items,
        total_items=len(action_items),
        source_word_count=sum(len(s.split()) for s in full_text_context),
    )
    
    run_context = NLPRunContext(
        task="action_items_assignee",
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
        metrics_structural={},
        gates={'low_coverage': transcript.metrics.coverage_ratio < 0.85},
    )
    
    artifact_path = output_dir / f"action_items_{alignment_path.stem.replace('alignment_', '')}.json"
    
    with open(artifact_path, 'w') as f:
        json.dump(artifact.to_dict(), f, indent=2)
        
    return artifact_path

