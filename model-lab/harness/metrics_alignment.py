"""
Alignment evaluation metrics.
Validates timestamp monotonicity and coverage.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

@dataclass
class AlignmentResult:
    """Container for Alignment evaluation results."""
    num_segments: int
    monotonicity_violations: int
    negative_duration_count: int
    coverage_ratio: float
    metadata: Dict[str, Any]

class AlignmentMetrics:
    """Calculate Alignment quality metrics."""

    @staticmethod
    def evaluate(segments: List[Dict[str, Any]],
                 audio_duration_s: float,
                 latency_s: float,
                 metadata: Dict[str, Any] = None) -> AlignmentResult:
        """
        Evaluate alignment/segmentation quality.
        
        Args:
            segments: List of {"start": float, "end": float, ...}
            audio_duration_s: Total audio duration
            latency_s: Processing time
        """
        num_segments = len(segments)
        monotonicity_violations = 0
        negative_duration_count = 0
        total_segment_duration = 0.0
        
        last_end = 0.0
        
        for seg in segments:
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            
            # Check duration
            duration = end - start
            if duration < 0:
                negative_duration_count += 1
                
            # Check monotonicity (start should be >= last_start? No, start >= last_end - overlap)
            # Strict monotonicity: start >= last_end
            # Allow some overlap? For "Alignment", overlap is usually bad if it's words.
            # But phrases might overlap in loose transcription.
            # We count violations of start < last_start (backtracking).
            if start < (last_end - 0.5): # Allow 500ms jitter/overlap?
                # Actually, check if start is BEFORE previous start
                pass 
            
            # Simple check: start < previous_end is not necessarily error (overlap).
            # But start[i] < start[i-1] is definitely error.
            # We can't check that easily in loop without history.
            
            total_segment_duration += max(0, duration)
            last_end = end

        # Check for backwards jumps
        for i in range(1, len(segments)):
            if segments[i]["start"] < segments[i-1]["start"]:
                monotonicity_violations += 1

        coverage = total_segment_duration / max(0.001, audio_duration_s)
        
        result = AlignmentResult(
            num_segments=num_segments,
            monotonicity_violations=monotonicity_violations,
            negative_duration_count=negative_duration_count,
            coverage_ratio=coverage,
            metadata={
                "latency_s": latency_s,
                **(metadata or {})
            }
        )
        
        logger.info(f"Alignment Eval: {num_segments} segs, Cov={coverage:.2f}, Violations={monotonicity_violations}")
        return result
