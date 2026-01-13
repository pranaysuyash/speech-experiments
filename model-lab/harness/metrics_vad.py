"""
VAD Metrics calculation.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class VADMetricsResult:
    speech_ratio: float
    num_segments: int
    avg_segment_duration: float
    failed_check: Optional[str] = None

class VADMetrics:
    @staticmethod
    def calculate(segments: List[Dict[str, int]], total_duration_samples: int, sr: int = 16000) -> VADMetricsResult:
        """
        Calculate VAD metrics.
        segments: list of {'start': int, 'end': int} in samples
        """
        if total_duration_samples <= 0:
            return VADMetricsResult(0.0, 0, 0.0, failed_check="zero_duration")
            
        total_speech_samples = sum(s['end'] - s['start'] for s in segments)
        speech_ratio = total_speech_samples / total_duration_samples
        
        num_segments = len(segments)
        avg_dur = (total_speech_samples / num_segments / sr) if num_segments > 0 else 0.0
        
        return VADMetricsResult(
            speech_ratio=speech_ratio,
            num_segments=num_segments,
            avg_segment_duration=avg_dur
        )

    @staticmethod
    def check_gates(metrics: VADMetricsResult, expectations: Dict[str, Any]) -> Dict[str, Any]:
        """Check if metrics meet expectations."""
        gates = {}
        
        # Check min speech ratio
        if "expected_speech_ratio_min" in expectations:
            gates["speech_ratio_min"] = metrics.speech_ratio >= expectations["expected_speech_ratio_min"]
            
        # Check max speech ratio
        if "expected_speech_ratio_max" in expectations:
            gates["speech_ratio_max"] = metrics.speech_ratio <= expectations["expected_speech_ratio_max"]
            
        # Check min segments
        if "expected_segments_min" in expectations:
            gates["segments_min"] = metrics.num_segments >= expectations["expected_segments_min"]
            
        # Check max segments
        if "expected_segments_max" in expectations:
            gates["segments_max"] = metrics.num_segments <= expectations["expected_segments_max"]
            
        # Overall pass logic
        gates["has_failure"] = any(v is False for v in gates.values())
        
        return gates
