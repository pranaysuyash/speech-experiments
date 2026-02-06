"""
Tests for action items with assignee runner.

Verifies:
1. Pass A / Pass B logic flow
2. Strict assignee validation (must be in allowed list)
3. Low coverage warnings
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from harness.alignment import AlignedSegment, AlignedTranscript, AlignmentMetrics
from scripts.run_action_items_with_assignee import extract_pass_b, run_action_items_with_assignee


class TestActionItemsAssignee(unittest.TestCase):
    def setUp(self):
        self.mock_alignment = AlignedTranscript(
            segments=[
                AlignedSegment(
                    start_s=0, end_s=10, text="I will fix the bug.", speaker_id="SPEAKER_01"
                ),
                AlignedSegment(start_s=10, end_s=20, text="Okay, thanks.", speaker_id="SPEAKER_02"),
            ],
            metrics=AlignmentMetrics(
                total_duration_s=20,
                assigned_duration_s=18,
                coverage_ratio=0.9,  # Good coverage
                unknown_ratio=0.1,
                speaker_switch_count=1,
                speaker_distribution={"SPEAKER_01": 10, "SPEAKER_02": 8},
            ),
            source_asr_path="asr.json",
            source_diarization_path="diar.json",
        )

    def test_pass_b_strict_assignee_validation(self):
        """Test that Pass B enforces allowed assignees."""
        candidates = [{"text": "Fix bug", "assignee": "SPEAKER_01", "evidence": "..."}]
        allowed = ["SPEAKER_01", "SPEAKER_02"]
        context = "..."

        # Mock LLM returning an hallucinated speaker
        mock_response = json.dumps(
            {
                "final_items": [
                    {"text": "Fix bug", "assignee": "SPEAKER_99", "priority": "high"},
                    {"text": "Review code", "assignee": "SPEAKER_02", "priority": "medium"},
                ]
            }
        )

        with patch("scripts.run_action_items_with_assignee.get_llm_completion") as mock_llm:
            mock_llm.return_value = MagicMock(success=True, text=mock_response)

            final_items = extract_pass_b(candidates, context, allowed, "model")

            # SPEAKER_99 should become UNKNOWN
            self.assertEqual(final_items[0]["assignee"], "UNKNOWN")
            # SPEAKER_02 is allowed, should stay
            self.assertEqual(final_items[1]["assignee"], "SPEAKER_02")

    @patch("scripts.run_action_items_with_assignee.load_alignment")
    @patch("scripts.run_action_items_with_assignee.extract_pass_a")
    @patch("scripts.run_action_items_with_assignee.extract_pass_b")
    @patch("scripts.run_action_items_with_assignee.compute_file_hash")
    @patch("builtins.open")
    @patch("json.dump")
    def test_runner_flow(
        self, mock_json_dump, mock_open, mock_compute_hash, mock_pass_b, mock_pass_a, mock_load
    ):
        """Test end-to-end runner flow."""
        mock_load.return_value = self.mock_alignment
        mock_compute_hash.return_value = "mock_hash"

        # Pass A finds one item
        mock_pass_a.return_value = [{"text": "Item 1", "assignee": "SPEAKER_01"}]
        # Pass B clarifies it
        mock_pass_b.return_value = [
            {"text": "Refined Item 1", "assignee": "SPEAKER_01", "priority": "high"}
        ]

        artifact, path = run_action_items_with_assignee(Path("alignment.json"))

        # Check artifact content
        self.assertEqual(len(artifact["output"]["action_items"]), 1)
        self.assertEqual(artifact["output"]["action_items"][0]["text"], "Refined Item 1")
        self.assertEqual(artifact["output"]["action_items"][0]["assignee"], "SPEAKER_01")

        # Check metrics
        self.assertEqual(
            artifact["metrics_structural"]["pass_a_count"], 2
        )  # Called for 2 speakers?
        # Actually items sum. extract_pass_a is called per speaker.
        # But here we mocked it. run_action_items calls extract_pass_a for each speaker.
        # The candidates list collects results.
        # Our mock setup is simplistic but verify pass_b was called with candidates.

        mock_pass_b.assert_called_once()

    @patch("scripts.run_action_items_with_assignee.load_alignment")
    @patch("scripts.run_action_items_with_assignee.extract_pass_a")
    @patch("scripts.run_action_items_with_assignee.extract_pass_b")
    @patch("scripts.run_action_items_with_assignee.compute_file_hash")
    @patch("builtins.open")
    @patch("json.dump")
    def test_low_coverage_gate(
        self, mock_json_dump, mock_open, mock_compute_hash, mock_pass_b, mock_pass_a, mock_load
    ):
        """Test that low coverage triggers gate flag."""
        # Set low coverage
        self.mock_alignment.metrics.coverage_ratio = 0.5
        mock_load.return_value = self.mock_alignment
        mock_compute_hash.return_value = "mock_hash"

        mock_pass_a.return_value = []
        mock_pass_b.return_value = []

        artifact, path = run_action_items_with_assignee(Path("alignment.json"))

        # Check gates
        self.assertTrue(artifact["gates"]["low_coverage"])


if __name__ == "__main__":
    unittest.main()
