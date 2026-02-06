"""
Integration test for SessionRunner using a real (small) audio input and resume.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness.media_ingest import IngestConfig
from harness.session import SessionRunner


class TestSessionRunnerE2E(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmp_dir.name)

        # Create a short valid WAV (requires ffmpeg for ingest step)
        import subprocess

        self.input_file = self.output_dir / "meeting.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(self.input_file),
            ],
            check=True,
            capture_output=True,
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_full_flow_and_resume_ingest_only(self):
        runner = SessionRunner(
            self.input_file,
            self.output_dir,
            preprocessing=IngestConfig(normalize=False, trim_silence=False),
            steps=["ingest"],
        )
        runner.run()

        run_dir = runner.session_dir
        manifest_path = run_dir / "manifest.json"
        self.assertTrue(manifest_path.exists())

        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(m["status"], "COMPLETED")
        self.assertIn("ingest", m["steps"])
        self.assertEqual(m["steps"]["ingest"]["status"], "COMPLETED")
        self.assertIn("bundle", m["steps"])
        self.assertEqual(m["steps"]["bundle"]["status"], "COMPLETED")

        bundle_manifest = run_dir / "bundle" / "bundle_manifest.json"
        self.assertTrue(bundle_manifest.exists())

        # Resume: ingest should be skipped, but bundle should still run.
        with patch("harness.session.ingest_media") as mock_ingest:
            runner2 = SessionRunner(
                self.input_file,
                self.output_dir,
                preprocessing=IngestConfig(normalize=False, trim_silence=False),
                steps=["ingest"],
                config={"resume_from": str(run_dir)},
            )
            runner2.run()
            mock_ingest.assert_not_called()


if __name__ == "__main__":
    unittest.main()
