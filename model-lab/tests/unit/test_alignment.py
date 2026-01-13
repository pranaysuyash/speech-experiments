"""
Tests for alignment module.

Tests verify:
1. Overlap calculation math
2. Segment alignment logic (best speaker, unknown handling)
3. Metrics calculation
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from harness.transcript_view import Segment
from harness.alignment import (
    calculate_overlap, align_segments, compute_metrics,
    AlignedSegment, AlignmentMetrics
)


class TestAlignmentMath(unittest.TestCase):
    """Test basic math functions."""
    
    def test_overlap_calculation(self):
        # Full overlap 1 inside 2
        self.assertEqual(calculate_overlap(1, 2, 0, 3), 1.0)
        # Full overlap 2 inside 1
        self.assertEqual(calculate_overlap(0, 3, 1, 2), 1.0)
        # Partial overlap (end of 1 overlaps start of 2)
        self.assertEqual(calculate_overlap(0, 2, 1, 3), 1.0)
        # No overlap
        self.assertEqual(calculate_overlap(0, 1, 2, 3), 0.0)
        # Touching boundaries (should be 0)
        self.assertEqual(calculate_overlap(0, 1, 1, 2), 0.0)


class TestSegmentAlignment(unittest.TestCase):
    """Test aligning ASR segments to turns."""
    
    def test_single_turn_overlap(self):
        """Segment largely overlapped by one speaker."""
        asr_segs = [Segment(start_s=0, end_s=5, text="Hello.")]
        turns = [
            {'start': 0, 'end': 5, 'speaker': 'SPEAKER_01'}
        ]
        
        aligned = align_segments(asr_segs, turns)
        
        self.assertEqual(len(aligned), 1)
        self.assertEqual(aligned[0].speaker_id, 'SPEAKER_01')
        self.assertEqual(aligned[0].confidence, 1.0)
        
    def test_multiple_turn_overlap_picks_max(self):
        """Segment overlaps two speakers, picking the one with more overlap."""
        # 0-5s segment, spk1: 0-4, spk2: 4-5
        asr_segs = [Segment(start_s=0, end_s=5, text="Hello.")]
        turns = [
            {'start': 0, 'end': 4, 'speaker': 'SPEAKER_01'},
            {'start': 4, 'end': 5, 'speaker': 'SPEAKER_02'}
        ]
        
        aligned = align_segments(asr_segs, turns)
        
        self.assertEqual(aligned[0].speaker_id, 'SPEAKER_01')
        # 4s overlap / 5s total = 0.8
        self.assertAlmostEqual(aligned[0].confidence, 0.8)

    def test_no_overlap_unknown(self):
        """No overlapping turns results in unknown."""
        asr_segs = [Segment(start_s=0, end_s=5, text="Hello.")]
        turns = [
            {'start': 10, 'end': 15, 'speaker': 'SPEAKER_01'}
        ]
        
        aligned = align_segments(asr_segs, turns)
        
        self.assertEqual(aligned[0].speaker_id, 'unknown')
        self.assertEqual(aligned[0].confidence, 0.0)
        
    def test_low_confidence_unknown(self):
        """Tiny overlap falls below threshold."""
        # 0.4s overlap on 10s segment = 0.04 < 0.1 threshold
        asr_segs = [Segment(start_s=0, end_s=10, text="Hello.")]
        turns = [
            {'start': 0, 'end': 0.4, 'speaker': 'SPEAKER_01'}
        ]
        
        aligned = align_segments(asr_segs, turns)
        
        self.assertEqual(aligned[0].speaker_id, 'unknown')


class TestMetrics(unittest.TestCase):
    """Test metric calculations."""
    
    def test_compute_metrics(self):
        segments = [
            AlignedSegment(start_s=0, end_s=10, text="", speaker_id="SPEAKER_01"),
            AlignedSegment(start_s=10, end_s=20, text="", speaker_id="SPEAKER_02"),
            AlignedSegment(start_s=20, end_s=30, text="", speaker_id="unknown"),
        ]
        
        metrics = compute_metrics(segments)
        
        self.assertEqual(metrics.total_duration_s, 30.0)
        self.assertEqual(metrics.assigned_duration_s, 20.0)
        self.assertAlmostEqual(metrics.coverage_ratio, 0.67, places=2)
        self.assertAlmostEqual(metrics.unknown_ratio, 0.33, places=2)
        self.assertEqual(metrics.speaker_switch_count, 1) # 01 -> 02
        self.assertEqual(metrics.speaker_distribution['SPEAKER_01'], 10.0)

if __name__ == '__main__':
    unittest.main()
