"""
Unit tests for Chapters (Semantic Segmentation).
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from harness.alignment import AlignedSegment
from harness.chapters import SemanticSegmenter


class TestChapters(unittest.TestCase):
    def setUp(self):
        # Mock SentenceTransformer to prevent download/load
        self.mock_model_class = patch("harness.chapters.SentenceTransformer").start()
        self.mock_model = MagicMock()
        self.mock_model_class.return_value = self.mock_model

        self.segmenter = SemanticSegmenter()

    def tearDown(self):
        patch.stopall()

    def test_window_creation(self):
        """Test sliding window text aggregation."""
        segments = [
            AlignedSegment(start_s=0, end_s=10, text="Hello", speaker_id="spk1", confidence=1.0),
            AlignedSegment(start_s=10, end_s=20, text="World", speaker_id="spk1", confidence=1.0),
            AlignedSegment(start_s=60, end_s=70, text="Later", speaker_id="spk1", confidence=1.0),
        ]

        windows = self.segmenter._create_windows(segments, duration=100.0)

        self.assertTrue(len(windows) > 0)
        self.assertIn("Hello", windows[0]["text"])
        self.assertIn("World", windows[0]["text"])

        # Verify stride
        self.assertEqual(windows[1]["start"], 15.0)

    def test_segmentation_logic(self):
        """Test boundary detection with mocked embeddings."""
        # Setup segments spanning enough time
        segments = []
        for i in range(10):  # 10 minutes approx?
            # Create segments every 10 seconds
            segments.append(
                AlignedSegment(
                    start_s=i * 60,
                    end_s=i * 60 + 10,
                    text=f"Seg {i}",
                    speaker_id="spk1",
                    confidence=1.0,
                )
            )

        # We need enough windows.
        # i goes 0..9. Times 0..540+.
        # Win 60, stride 15.

        # Mock embeddings
        # Let's say windows 0-5 are Topic A, 6-10 are Topic B.
        # Topic A vec: [1, 0]
        # Topic B vec: [0, 1]

        def mock_encode(texts):
            for _t in texts:
                # Naive: if "Seg 0" to "Seg 3" -> Topic A
                # If "Seg 4" starts appearing -> transition
                # Real logic in test is determining how `_create_windows` groups them.
                # Let's simple return random distinct vectors based on index to force boundaries?
                # Or just fixed list.
                pass
            # Better: just return a fixed array of shape (N, 2)
            n = len(texts)
            arr = np.zeros((n, 2))

            # First half Topic A
            cutoff = n // 2
            arr[:cutoff, 0] = 1.0
            arr[cutoff:, 1] = 1.0
            return arr

        self.mock_model.encode.side_effect = mock_encode

        chapters, config = self.segmenter.segment(segments, duration=600.0)

        # We expect at least 2 chapters (approx)
        self.assertTrue(len(chapters) >= 2)
        self.assertIn("threshold_value", config)

        # Check continuity
        self.assertEqual(chapters[0].start, 0.0)
        self.assertEqual(chapters[-1].end, 600.0)
        self.assertEqual(chapters[0].end, chapters[1].start)

    def test_empty_segments(self):
        # 0.0 is short -> returns fallback chapter
        chapters, config = self.segmenter.segment([], 0.0)
        self.assertEqual(len(chapters), 1)
        self.assertEqual(config["gating"], "short_duration")

    def test_short_duration(self):
        chapters, config = self.segmenter.segment([], 10.0)
        self.assertEqual(len(chapters), 1)
        self.assertEqual(config["gating"], "short_duration")
