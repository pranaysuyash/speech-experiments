"""
Unit tests for SessionRunner logic.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from harness.session import SessionRunner

class TestSessionRunner(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmp_dir.name)
        
        # Mock input
        self.input_file = self.output_dir / "input.wav"
        self.input_file.write_text("dummy audio content")
        
    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_init_creates_directories(self):
        runner = SessionRunner(self.input_file, self.output_dir)
        runner._init_dirs()
        
        self.assertTrue(runner.session_dir.exists())
        self.assertTrue(runner.ctx.artifacts_dir.exists())
        self.assertTrue((runner.session_dir / "manifest.json").parent.exists())

    def test_resume_skip_logic(self):
        runner = SessionRunner(self.input_file, self.output_dir)
        runner._init_dirs()
        
        # Calculate expected hash for validation
        expected_hash = runner.input_hash
        
        # Manually create a completed step in manifest
        runner.manifest = runner._default_manifest() # Ensure manifest is dict
        runner.manifest["steps"]["step1"] = {
            "status": "COMPLETED",
            "artifacts": [{"type": "json", "path": "artifacts/step1.json", "hash": "abc"}]
        }
        artifact_path = runner.ctx.artifacts_dir / "step1.json"
        
        # Mock step def
        step_def = MagicMock()
        step_def.name = "step1"
        # artifact_paths is used inside _is_step_valid_for_resume to verify artifacts?
        # No, _is_step_valid_for_resume calls _artifact_exists_and_hash_matches for each artifact in manifest entry.
        # It does NOT use step_def.artifact_paths unless checking something else?
        # Let's check implementation. It iterates entry["artifacts"].
        
        # Test 1: Artifact missing -> Should NOT be valid (return False)
        self.assertFalse(runner._is_step_valid_for_resume(runner.manifest, "step1", step_def))
        
        # Test 2: Artifact exists but matching hash missing -> Should NOT be valid
        # (Our mock manifest has hash "abc")
        artifact_path.touch()
        with open(artifact_path, 'w') as f:
             f.write("content") # hash will not be 'abc'
        self.assertFalse(runner._is_step_valid_for_resume(runner.manifest, "step1", step_def))
        
        # Test 3: Artifact exists AND hash matches -> Should be valid (return True)
        # We need to compute hash of "content" or update manifest to match.
        from harness.media_ingest import sha256_file
        real_hash = sha256_file(artifact_path)
        runner.manifest["steps"]["step1"]["artifacts"][0]["hash"] = real_hash
        
        self.assertTrue(runner._is_step_valid_for_resume(runner.manifest, "step1", step_def))

    def test_crash_recovery_steps(self):
        runner = SessionRunner(self.input_file, self.output_dir)
        runner._init_dirs()
        
        # Create a manifest with a STALE RUNNING step
        manifest = {
            "status": "RUNNING",
            "started_at": "old_timestamp",
            "steps": {
                "ingest": {"status": "COMPLETED", "result": {}, "artifacts": []},
                "asr": {"status": "RUNNING", "started_at": "old_timestamp"}
            }
        }
        (runner.session_dir / "manifest.json").write_text(json.dumps(manifest))
        
        # Initialize NEW session resuming from this dir
        runner2 = SessionRunner(self.input_file, self.output_dir)
        runner2.run_id = runner.run_id
        runner2.session_dir = runner.session_dir
        runner2.manifest_path = runner.manifest_path
        
        # We also need to mock _topo_order and execution to avoid running real stuff
        with patch.object(runner2, "_execute_step") as mock_exec, \
             patch.object(runner2, "_topo_order", return_value=["asr"]), \
             patch.object(runner2, "_export_partial_bundle"):
             
            runner2.run()
            
            # Verify manifest was updated (reloaded from disk)
            m = runner2._load_manifest()
            self.assertEqual(m["steps"]["asr"]["status"], "FAILED")
            self.assertEqual(m["steps"]["asr"]["error"]["type"], "StaleRun")
            
            # Verify it proceeded to execute (since it was FAILED/not COMPLETED)
            mock_exec.assert_called()

    def test_eval_json_is_written_at_run_root(self):
        runner = SessionRunner(self.input_file, self.output_dir)

        with patch.object(runner, "_execute_step"), \
             patch.object(runner, "_topo_order", return_value=[]), \
             patch.object(runner, "_export_partial_bundle"), \
             patch("harness.meeting_pack.build_meeting_pack"):
            runner.run()

        eval_path = runner.session_dir / "eval.json"
        self.assertTrue(eval_path.exists())
        data = json.loads(eval_path.read_text())
        self.assertEqual(data["schema_version"], "1")
        self.assertEqual(data["run_id"], runner.run_id)

if __name__ == "__main__":
    unittest.main()
