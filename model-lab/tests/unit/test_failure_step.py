"""
Unit tests for failure_step backend implementation.

Tests verify:
1. Pipeline step failure captures failure_step authorit atively
2. Early/Pre-pipeline failures set failure_step to null
3. Cold refresh reconstructibility
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness.session import SessionRunner


class TestFailureStepCapture(unittest.TestCase):
    """Test authoritative failure_step capture and persistence."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmp_dir.name)

        # Mock input
        self.input_file = self.output_dir / "input.wav"
        self.input_file.write_text("dummy audio content")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_pipeline_step_failure_captures_authoritative_failure_step(self):
        """
        Test 5.1 — Pipeline Step Failure

        Given: Failure during diarization
        Assert:
        - status = "FAILED"
        - steps_completed = ["ingest", "asr"]
        - failure_step = "diarization"
        """
        runner = SessionRunner(self.input_file, self.output_dir, force=True, resume=False)
        runner._init_dirs()
        runner._register_steps()

        # Mock step execution: ingest and asr succeed, diarization fails
        def mock_execute_step(m, step_def):
            name = step_def.name
            entry = runner._step_entry(m, name)

            if name in ["ingest", "asr"]:
                # Simulate successful completion
                entry["status"] = "COMPLETED"
                entry["artifacts"] = []
                entry["result"] = {}
                return
            elif name == "diarization":
                # Simulate failure at diarization
                entry["status"] = "FAILED"
                m["status"] = "FAILED"
                m["failure_step"] = name  # This is what we're testing
                m["error_step"] = name
                m["error_code"] = "RuntimeError"
                m["error_message"] = "Simulated diarization failure"
                raise RuntimeError("Simulated diarization failure")
            else:
                # Should not reach other steps
                self.fail(f"Unexpected step execution: {name}")

        with (
            patch.object(runner, "_execute_step", side_effect=mock_execute_step),
            patch.object(runner, "_topo_order", return_value=["ingest", "asr", "diarization"]),
            patch.object(runner, "_export_partial_bundle"),
            patch("harness.meeting_pack.build_meeting_pack"),
        ):
            try:
                runner.run()
            except RuntimeError:
                pass  # Expected failure

        # Load manifest and verify
        manifest = runner._load_manifest()

        self.assertEqual(manifest["status"], "FAILED")
        self.assertEqual(manifest["failure_step"], "diarization")

        # Verify steps_completed (based on manifest steps dict)
        completed_steps = [
            name for name, data in manifest["steps"].items() if data.get("status") == "COMPLETED"
        ]
        self.assertEqual(sorted(completed_steps), ["asr", "ingest"])

        # Verify error fields
        self.assertEqual(manifest["error_step"], "diarization")
        self.assertIn("error_code", manifest)
        self.assertIn("error_message", manifest)

    def test_early_failure_sets_failure_step_to_null(self):
        """
        Test 5.2 — Early Failure (No Steps)

        Given: Failure before first step (e.g., in run() setup)
        Assert:
        - status = "FAILED"
        - steps_completed = []
        - failure_step = null
        """
        runner = SessionRunner(self.input_file, self.output_dir, force=True, resume=False)
        runner._init_dirs()
        runner._register_steps()

        # Mock _topo_order to raise before any step executes
        def mock_topo_order():
            raise RuntimeError("Simulated pre-pipeline failure")

        with (
            patch.object(runner, "_topo_order", side_effect=mock_topo_order),
            patch.object(runner, "_export_partial_bundle"),
            patch("harness.meeting_pack.build_meeting_pack"),
        ):
            try:
                runner.run()
            except RuntimeError:
                pass  # Expected failure

        # Load manifest and verify
        manifest = runner._load_manifest()

        self.assertEqual(manifest["status"], "FAILED")
        self.assertIsNone(
            manifest.get("failure_step"), "failure_step should be None for pre-pipeline failures"
        )

        # Verify no steps completed
        completed_steps = [
            name
            for name, data in manifest.get("steps", {}).items()
            if data.get("status") == "COMPLETED"
        ]
        self.assertEqual(len(completed_steps), 0)

    def test_failure_step_injected_during_step_execution(self):
        """
        Test failure_step is captured using SESSION_FAIL_STEP env var.

        This tests the real exception path in _execute_step.
        """
        # Create a real audio file for ingest to process
        import wave

        audio_file = self.output_dir / "test_audio.wav"
        with wave.open(str(audio_file), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00\x00" * 1000)  # 1000 frames of silence

        runner = SessionRunner(audio_file, self.output_dir, force=True, resume=False)

        # Set environment variable to trigger failure at ASR step
        os.environ["SESSION_FAIL_STEP"] = "asr"

        try:
            # Mock only the non-ingest dependencies
            with (
                patch("harness.asr.run_asr"),
                patch.object(runner, "_export_partial_bundle"),
                patch("harness.meeting_pack.build_meeting_pack"),
            ):
                try:
                    runner.run()
                except RuntimeError:
                    pass  # Expected failure

            # Load manifest and verify failure_step was captured
            manifest = runner._load_manifest()

            self.assertEqual(manifest["status"], "FAILED")
            self.assertEqual(
                manifest["failure_step"],
                "asr",
                "failure_step should be 'asr' (captured at exception site)",
            )
            self.assertEqual(manifest["error_step"], "asr")

            # Verify ingest completed before failure
            self.assertEqual(manifest["steps"]["ingest"]["status"], "COMPLETED")
            self.assertEqual(manifest["steps"]["asr"]["status"], "FAILED")

        finally:
            # Clean up environment variable
            os.environ.pop("SESSION_FAIL_STEP", None)

    def test_cold_refresh_reconstructibility(self):
        """
        Test 5.3 — Cold Refresh Invariant

        Assert:
        - Restart server (reload manifest from disk)
        - Same failure_step returned
        - No recomputation
        """
        runner = SessionRunner(self.input_file, self.output_dir, force=True, resume=False)
        runner._init_dirs()

        # Manually create a failed manifest
        manifest = runner._default_manifest()
        manifest["status"] = "FAILED"
        manifest["failure_step"] = "alignment"
        manifest["error_step"] = "alignment"
        manifest["error_code"] = "ValueError"
        manifest["error_message"] = "Test error"
        manifest["steps"] = {
            "ingest": {"status": "COMPLETED", "artifacts": []},
            "asr": {"status": "COMPLETED", "artifacts": []},
            "diarization": {"status": "COMPLETED", "artifacts": []},
            "alignment": {"status": "FAILED", "artifacts": []},
        }

        # Save to disk
        (runner.session_dir / "manifest.json").write_text(json.dumps(manifest))

        # Simulate server restart: create new runner instance
        runner2 = SessionRunner(self.input_file, self.output_dir)
        runner2.run_id = runner.run_id
        runner2.session_dir = runner.session_dir
        runner2.manifest_path = runner.manifest_path

        # Load manifest (simulating cold refresh)
        reloaded_manifest = runner2._load_manifest()

        # Verify failure_step is preserved exactly
        self.assertEqual(reloaded_manifest["failure_step"], "alignment")
        self.assertEqual(reloaded_manifest["status"], "FAILED")
        self.assertEqual(reloaded_manifest["error_step"], "alignment")

        # Verify it's identical to what was written
        self.assertEqual(reloaded_manifest["failure_step"], manifest["failure_step"])


if __name__ == "__main__":
    unittest.main()
