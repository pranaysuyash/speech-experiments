"""
NLP Artifact Schema - Structured outputs for audio-derived NLP tasks.

These tasks consume ASR artifacts and produce structured outputs.
No freeform text generation - all outputs are schema-constrained.

Key provenance fields:
- parent_artifact_path: Path to source artifact (e.g., ASR result)
- parent_artifact_hash: Hash of parent artifact for reproducibility
- nlp_model_id: Model used for NLP processing
- prompt_hash: Hash of prompt template used

Quality metrics are None unless labeled datasets exist.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SummaryOutput:
    """Constrained summary output - no freeform generation."""
    sentences: List[str]              # Individual summary sentences
    total_sentences: int              # Count for validation
    source_word_count: int            # Words in source transcript
    compression_ratio: float          # source_words / summary_words
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sentences': self.sentences,
            'total_sentences': self.total_sentences,
            'source_word_count': self.source_word_count,
            'compression_ratio': round(self.compression_ratio, 2),
        }


@dataclass
class EvidenceSpan:
    """Evidence span from transcript supporting an extraction."""
    quote: str                         # Exact quote from transcript
    start_word: Optional[int] = None   # Start word index
    end_word: Optional[int] = None     # End word index
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'quote': self.quote,
            'start_word': self.start_word,
            'end_word': self.end_word,
        }


@dataclass
class ActionItem:
    """Extracted action item from meeting."""
    text: str                          # Action item text (required, non-empty)
    assignee: Optional[str] = None     # Who should do it
    due: Optional[str] = None          # When (ISO format or description)
    priority: Optional[str] = None     # high/medium/low
    evidence: List[Dict] = field(default_factory=list)  # Source spans
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'assignee': self.assignee,
            'due': self.due,
            'priority': self.priority,
            'evidence': self.evidence,
        }


@dataclass
class ActionItemsOutput:
    """Constrained action items output with operability thresholds."""
    action_items: List[ActionItem]
    total_items: int
    source_word_count: int
    
    # Operability thresholds
    MAX_ITEMS = 20
    MIN_TEXT_LENGTH = 5
    
    def validate(self) -> List[str]:
        """Validate operability thresholds. Returns list of violations."""
        violations = []
        
        if len(self.action_items) > self.MAX_ITEMS:
            violations.append(f"Too many items: {len(self.action_items)} > {self.MAX_ITEMS}")
        
        for i, item in enumerate(self.action_items):
            if not item.text or len(item.text.strip()) < self.MIN_TEXT_LENGTH:
                violations.append(f"Item {i}: text too short or empty")
            if len(item.text) > 500:
                violations.append(f"Item {i}: text too long ({len(item.text)} chars)")
        
        return violations
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action_items': [item.to_dict() for item in self.action_items],
            'total_items': self.total_items,
            'source_word_count': self.source_word_count,
        }


@dataclass
class Entity:
    """Named entity from transcript."""
    text: str                          # Entity text
    type: str                          # PERSON, ORG, DATE, etc.
    start_char: Optional[int] = None   # Start position in transcript
    end_char: Optional[int] = None     # End position
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'type': self.type,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'confidence': round(self.confidence, 2),
        }


@dataclass
class NLPRunContext:
    """Run context for NLP tasks."""
    task: str                          # summarize, action_items, ner
    nlp_model_id: str                  # Model used for NLP
    timestamp: str
    git_hash: Optional[str] = None
    command: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task': self.task,
            'nlp_model_id': self.nlp_model_id,
            'timestamp': self.timestamp,
            'git_hash': self.git_hash,
            'command': self.command,
        }


@dataclass
class NLPInputs:
    """Inputs for NLP task - provenance linkage."""
    parent_artifact_path: str          # Path to source artifact
    parent_artifact_hash: str          # SHA256 of parent artifact file
    asr_model_id: str                  # ASR model that produced transcript
    asr_text_hash: str                 # Hash of transcript text
    transcript_word_count: int
    audio_duration_s: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'parent_artifact_path': self.parent_artifact_path,
            'parent_artifact_hash': self.parent_artifact_hash,
            'asr_model_id': self.asr_model_id,
            'asr_text_hash': self.asr_text_hash,
            'transcript_word_count': self.transcript_word_count,
            'audio_duration_s': self.audio_duration_s,
        }


@dataclass
class NLPProvenance:
    """Provenance for NLP processing."""
    prompt_template: str               # Template used (not the filled prompt)
    prompt_hash: str                   # Hash of template
    has_ground_truth: bool = False     # Usually false
    metrics_valid: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'prompt_template': self.prompt_template,
            'prompt_hash': self.prompt_hash,
            'has_ground_truth': self.has_ground_truth,
            'metrics_valid': self.metrics_valid,
        }


@dataclass 
class NLPArtifact:
    """
    Artifact for audio-derived NLP tasks.
    
    Links back to parent ASR artifact for full provenance chain.
    """
    run_context: NLPRunContext
    inputs: NLPInputs
    provenance: NLPProvenance
    output: Dict[str, Any]             # Task-specific output (SummaryOutput, etc.)
    metrics_structural: Dict[str, Any]  # Latency, token counts, etc.
    gates: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'run_context': self.run_context.to_dict(),
            'inputs': self.inputs.to_dict(),
            'provenance': self.provenance.to_dict(),
            'output': self.output,
            'metrics_structural': self.metrics_structural,
            'gates': self.gates,
            'errors': self.errors,
        }


def compute_text_hash(text: str, length: int = 16) -> str:
    """Compute hash of text content."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:length]


def compute_file_hash(path: Path, length: int = 16) -> str:
    """Compute hash of file contents."""
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:length]


def load_asr_artifact(path: Path) -> Dict[str, Any]:
    """Load and validate an ASR artifact."""
    if not path.exists():
        raise FileNotFoundError(f"ASR artifact not found: {path}")
    
    with open(path) as f:
        artifact = json.load(f)
    
    # Validate it's an ASR artifact
    task = artifact.get('run_context', {}).get('task')
    if task != 'asr':
        raise ValueError(f"Expected ASR artifact, got task='{task}'")
    
    # Extract transcript
    output = artifact.get('output', {})
    transcript = output.get('text', '')
    
    if not transcript:
        raise ValueError("ASR artifact has no transcript text")
    
    return artifact


def validate_nlp_artifact(artifact: NLPArtifact) -> None:
    """Validate NLP artifact structure."""
    # Parent linkage required
    if not artifact.inputs.parent_artifact_hash:
        raise ValueError("NLP artifact must have parent_artifact_hash")
    
    if not artifact.inputs.asr_text_hash:
        raise ValueError("NLP artifact must have asr_text_hash")
    
    # Prompt provenance required
    if not artifact.provenance.prompt_hash:
        raise ValueError("NLP artifact must have prompt_hash")


# Prompt templates for reproducibility
SUMMARY_PROMPT_TEMPLATE = """Summarize the following transcript in {max_sentences} sentences or fewer.
Focus on key points, decisions, and outcomes.
Do not add information not present in the transcript.

Transcript:
{transcript}

Summary (bullet points):"""

ACTION_ITEMS_PROMPT_TEMPLATE = """Extract action items from the following meeting transcript.
Return ONLY valid JSON in this exact format:
{
  "action_items": [
    {
      "text": "Brief description of what needs to be done",
      "assignee": "Person name or null",
      "due": "Date/deadline or null",
      "priority": "high" | "medium" | "low" | null
    }
  ]
}

Rules:
- Only include explicit action items, not general discussion
- Maximum 20 items
- Each item text must be 5-500 characters
- If no action items found, return {"action_items": []}

Transcript:
{transcript}

JSON response:"""

