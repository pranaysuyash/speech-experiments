"""
Speaker Alignment Module - Map ASR segments to Diarization turns.

Provides logic to attribute a speaker to each ASR segment based on temporal overlap
with diarization outputs. This enables "who said what" features (attribution).

Usage:
    from harness.alignment import align_artifacts
    
    alignment = align_artifacts(asr_path, diarization_path)
    for seg in alignment.segments:
        print(f"[{seg.start_s}-{seg.end_s}] {seg.speaker_id}: {seg.text}")
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from harness.transcript_view import TranscriptView, Segment, from_asr_artifact

logger = logging.getLogger("alignment")


@dataclass
class AlignedSegment(Segment):
    """ASR segment with assigned speaker."""
    speaker_id: str = "unknown"
    confidence: float = 0.0  # Overlap ratio (0.0 to 1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'speaker_id': self.speaker_id,
            'confidence': round(self.confidence, 2),
        })
        return d


@dataclass
class AlignmentMetrics:
    """Metrics for alignment quality."""
    total_duration_s: float
    assigned_duration_s: float
    coverage_ratio: float  # assigned / total
    unknown_ratio: float
    speaker_switch_count: int
    speaker_distribution: Dict[str, float]  # speaker_id -> duration_s
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_duration_s': round(self.total_duration_s, 2),
            'assigned_duration_s': round(self.assigned_duration_s, 2),
            'coverage_ratio': round(self.coverage_ratio, 2),
            'unknown_ratio': round(self.unknown_ratio, 2),
            'speaker_switch_count': self.speaker_switch_count,
            'speaker_distribution': {
                k: round(v, 2) for k, v in self.speaker_distribution.items()
            },
        }


@dataclass
class AlignedTranscript:
    """Result of alignment process."""
    segments: List[AlignedSegment]
    metrics: AlignmentMetrics
    source_asr_path: str
    source_diarization_path: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'segments': [s.to_dict() for s in self.segments],
            'metrics': self.metrics.to_dict(),
            'source_asr_path': self.source_asr_path,
            'source_diarization_path': self.source_diarization_path,
        }


def calculate_overlap(
    start1: float, end1: float, 
    start2: float, end2: float
) -> float:
    """Calculate overlap duration between two time ranges."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    return max(0.0, overlap_end - overlap_start)


def align_segments(
    asr_segments: List[Segment], 
    diarization_turns: List[Dict]
) -> List[AlignedSegment]:
    """
    Align ASR segments with diarization turns.
    
    Strategy:
    For each ASR segment, find overlapping diarization turns.
    Assign the speaker with the maximum overlap duration.
    """
    aligned_segments = []
    
    # Sort turns by start time for efficiency (though usually already sorted)
    diarization_turns.sort(key=lambda x: x.get('start', 0))
    
    for seg in asr_segments:
        # Find all overlapping turns
        overlaps = {}  # speaker -> total_overlap_duration
        
        # Optimization: Use binary search or sliding window if needed for huge files
        # For now, linear scan with start-time prune is okay-ish but simple scan is safest
        # Actually, let's do a simple optimization: only check possible candidates
        
        candidates = [
            turn for turn in diarization_turns
            if turn.get('end', 0) > seg.start_s and turn.get('start', 0) < seg.end_s
        ]
        
        for turn in candidates:
            speaker = turn.get('speaker', 'unknown')
            overlap = calculate_overlap(
                seg.start_s, seg.end_s,
                turn.get('start', 0), turn.get('end', 0)
            )
            
            if speaker not in overlaps:
                overlaps[speaker] = 0.0
            overlaps[speaker] += overlap
        
        # Assign best speaker
        if not overlaps:
            best_speaker = "unknown"
            confidence = 0.0
        else:
            best_speaker = max(overlaps, key=overlaps.get)
            total_overlap = overlaps[best_speaker]
            segment_duration = seg.end_s - seg.start_s
            confidence = total_overlap / segment_duration if segment_duration > 0 else 0.0
            
            # Clamp confidence
            confidence = min(1.0, confidence)
            
            # If overlap is tiny, maybe mark unknown? (e.g. < 10%)
            if confidence < 0.1:
                best_speaker = "unknown"
                confidence = 0.0
        
        aligned_segments.append(AlignedSegment(
            start_s=seg.start_s,
            end_s=seg.end_s,
            text=seg.text,
            speaker=seg.speaker, # Preserve existing if any (mostly None)
            speaker_id=best_speaker,
            confidence=confidence,
        ))
        
    return aligned_segments


def compute_metrics(segments: List[AlignedSegment]) -> AlignmentMetrics:
    """Compute metrics for the alignment."""
    total_dur = sum(s.duration_s for s in segments)
    assigned_dur = sum(s.duration_s for s in segments if s.speaker_id != "unknown")
    
    speaker_dist = {}
    last_speaker = None
    switch_count = 0
    
    for s in segments:
        if s.speaker_id != "unknown":
            speaker_dist[s.speaker_id] = speaker_dist.get(s.speaker_id, 0) + s.duration_s
            
            if last_speaker is not None and s.speaker_id != last_speaker:
                switch_count += 1
            last_speaker = s.speaker_id
            
    return AlignmentMetrics(
        total_duration_s=total_dur,
        assigned_duration_s=assigned_dur,
        coverage_ratio=assigned_dur / total_dur if total_dur > 0 else 0,
        unknown_ratio=1.0 - (assigned_dur / total_dur if total_dur > 0 else 0),
        speaker_switch_count=switch_count,
        speaker_distribution=speaker_dist,
    )


def align_artifacts(asr_path: str, diarization_path: str) -> AlignedTranscript:
    """
    Load artifacts and perform alignment.
    """
    # Load ASR
    asr_view = from_asr_artifact(Path(asr_path))
    
    # Load Diarization
    with open(diarization_path) as f:
        diar_artifact = json.load(f)
        
    diar_turns = diar_artifact.get('output', {}).get('segments', [])
    
    # Align
    aligned_segments = align_segments(asr_view.segments, diar_turns)
    
    # Metrics
    metrics = compute_metrics(aligned_segments)
    
    return AlignedTranscript(
        segments=aligned_segments,
        metrics=metrics,
        source_asr_path=str(asr_path),
        source_diarization_path=str(diarization_path),
    )


def load_alignment(path: Path) -> AlignedTranscript:
    """Load alignment artifact from disk."""
    with open(path) as f:
        data = json.load(f)
    
    # Handle wrapped artifact "output" vs raw keys
    output = data.get("output", data)
    raw_segments = output.get("segments", [])
    
    segments = []
    for s in raw_segments:
        segments.append(AlignedSegment(
            start_s=s.get("start_s", s.get("start", 0.0)),
            end_s=s.get("end_s", s.get("end", 0.0)),
            text=s.get("text", ""),
            speaker=s.get("speaker"),
            speaker_id=s.get("speaker_id", "unknown"),
            confidence=s.get("confidence", 0.0)
        ))
        
    metrics_data = output.get("metrics", {})
    metrics = AlignmentMetrics(
        total_duration_s=metrics_data.get("total_duration_s", 0.0),
        assigned_duration_s=metrics_data.get("assigned_duration_s", 0.0),
        coverage_ratio=metrics_data.get("coverage_ratio", 0.0),
        unknown_ratio=metrics_data.get("unknown_ratio", 0.0),
        speaker_switch_count=metrics_data.get("speaker_switch_count", 0),
        speaker_distribution=metrics_data.get("speaker_distribution", {})
    )
    
    inputs = data.get("inputs", {})
    return AlignedTranscript(
        segments=segments,
        metrics=metrics,
        source_asr_path=inputs.get("parent_artifact_path", inputs.get("parent_artifact_hash", "")),
        source_diarization_path=""
    )


def run_alignment(
    asr_path: Path,
    diarization_path: Path,
    output_dir: Path,
    force: bool = False
) -> Path:
    """
    Run alignment task.
    
    Args:
        asr_path: Path to ASR artifact.
        diarization_path: Path to Diarization artifact.
        output_dir: Directory to save artifact.
        force: Overwrite existing.
        
    Returns:
        Path to alignment artifact.
    """
    from harness.runner_schema import compute_pcm_hash, compute_file_hash
    
    asr_path = asr_path.resolve()
    diarization_path = diarization_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check inputs exist
    if not asr_path.exists():
        raise FileNotFoundError(f"ASR not found: {asr_path}")
    if not diarization_path.exists():
        raise FileNotFoundError(f"Diarization not found: {diarization_path}")
        
    # Run Core Logic
    aligned_transcript = align_artifacts(str(asr_path), str(diarization_path))
    
    # Save Artifact
    artifact_name = f"alignment_{asr_path.stem.replace('asr_', '')}.json"
    artifact_path = output_dir / artifact_name
    
    # Compute hashes
    asr_hash = compute_file_hash(asr_path)
    diar_hash = compute_file_hash(diarization_path)
    
    # Construct output JSON (wrapping core logic result)
    output_data = {
        "segments": [s.to_dict() for s in aligned_transcript.segments],
        "metrics": aligned_transcript.metrics.to_dict(),
        "source_asr_path": str(asr_path),
        "source_diarization_path": str(diarization_path)
    }
    
    final_data = {
        "inputs": {
            "parent_artifact_path": str(asr_path),
            "parent_artifact_hash": asr_hash,
            "diarization_path": str(diarization_path),
            "diarization_hash": diar_hash
        },
        "output": output_data
    }
    
    with open(artifact_path, 'w') as f:
        json.dump(final_data, f, indent=2)
        
    return artifact_path

