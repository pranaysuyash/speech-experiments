import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import UTC

from harness.session import SessionContext

# Import the function to test.
# Note: We need to patch run_alignment inside harness.session because it's imported there.
from server.api.runs import get_run_status


class TestBackendFixes(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_runs_root = os.environ.get("MODEL_LAB_RUNS_ROOT")
        os.environ["MODEL_LAB_RUNS_ROOT"] = self.test_dir
        self.runs_root = Path(self.test_dir)

        # Create a dummy input file for SessionRunner init
        self.dummy_input = self.runs_root / "dummy_input.wav"
        with open(self.dummy_input, "wb") as f:
            f.write(b"dummy audio content")

        # Initialize common SessionRunner
        from harness.session import SessionRunner

        self.runner = SessionRunner(
            input_path=self.dummy_input, output_dir=self.runs_root, force=True
        )
        self.runner._register_steps()

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        if self.original_runs_root:
            os.environ["MODEL_LAB_RUNS_ROOT"] = self.original_runs_root
        else:
            del os.environ["MODEL_LAB_RUNS_ROOT"]

    def test_alignment_func_contract(self):
        """
        Verify that alignment_func returns a dictionary with 'artifacts' list,
        not a raw Path object.
        """
        print("\n[Test] Verifying alignment_func contract (PosixPath fix)...")
        # Use common runner
        runner = self.runner
        alignment_step_def = runner.steps["alignment"]
        alignment_func = alignment_step_def.func

        # Mock context (needs artifacts_dir and some helper methods)
        ctx = MagicMock(spec=SessionContext)
        ctx.artifacts_dir = Path(self.test_dir) / "artifacts"
        ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # We also need to mock get_artifact (new method) to avoid actual lookups
        runner.get_artifact = MagicMock(return_value=ctx.artifacts_dir / "asr.json")

        # Ensure fallback paths exist so checks inside alignment_func pass
        (ctx.artifacts_dir / "asr.json").touch()
        (ctx.artifacts_dir / "diarization.json").touch()

        # Mock run_alignment to return a Path object, simulating the actual behavior
        mock_path = Path("/tmp/some/artifact.rttm")

        # We need to patch where it is defined, not where it is imported inside a function
        with patch("harness.alignment.run_alignment", return_value=mock_path) as mock_run:
            result = alignment_func(ctx)

            # Verification 1: It should return a dict
            self.assertIsInstance(
                result, dict, f"alignment_func should return a dict, got {type(result)}"
            )

            # Verification 2: It should have 'artifacts' key
            self.assertIn("artifacts", result, "Result dict missing 'artifacts' key")

            # Verification 3: Artifacts should be a list of dicts
            artifacts = result["artifacts"]
            self.assertIsInstance(artifacts, list, "artifacts should be a list")
            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0]["path"], str(mock_path))

            print("✅ alignment_func correctly wraps Path in dictionary.")

    def test_stale_run_snapshot_consistency(self):
        """
        Verify that a STALE run returns the full manifest data, not just a heartbeat status.
        """
        print("\n[Test] Verifying STALE run snapshot consistency...")

        from datetime import datetime, timedelta

        # Setup a dummy run with correct structure: sessions/<hash>/<run_id>
        run_id = "test_run_stale"
        input_hash = "dummy_hash"
        run_dir = Path(self.test_dir) / "sessions" / input_hash / run_id
        run_dir.mkdir(parents=True)

        # Create a manifest with specific data we want to check for
        # Set updated_at to > 90s ago
        old_time = (
            (datetime.now(UTC) - timedelta(seconds=200)).isoformat().replace("+00:00", "Z")
        )

        manifest_data = {
            "run_id": run_id,
            "status": "RUNNING",  # It says RUNNING on disk, but updated_at will make it STALE
            "updated_at": old_time,
            "steps": {
                "ingest": {"status": "COMPLETED"},
                "asr": {"status": "COMPLETED"},
                "alignment": {"status": "RUNNING"},
            },
            "current_step": "alignment",
            "failure_step": "some_failed_step",  # Should be returned if present
            "meta": {"important_field": "must_be_present"},
            "input_path": "dummy.wav",
        }
        with open(run_dir / "manifest.json", "w") as f:
            json.dump(manifest_data, f)

        # We don't need heartbeat file, the API checks manifests updated_at

        # We need to force the index to refresh so it picks up our new file
        from server.services.runs_index import get_index

        get_index().refresh()

        # Call the API function
        status_response = get_run_status(run_id)

        # Verification 1: Status should be STALE (calculated dynamically)
        self.assertEqual(status_response["status"], "STALE", "Run should be detected as STALE")

        # Verification 2: Manifest data must be present (The Fix)
        # We check fields that are projected from the manifest
        self.assertEqual(status_response["current_step"], "alignment")
        self.assertEqual(status_response["failure_step"], "some_failed_step")
        self.assertIsNotNone(status_response["meta"]["manifest_mtime"])
        # Authoritative list check (keys of steps)
        self.assertIn("ingest", status_response["steps_completed"])
        self.assertIn("asr", status_response["steps_completed"])

        print("✅ STALE run correctly returns full manifest data.")

    # --- Phase 2 Hardening Tests ---

    def test_strict_normalizer(self):
        """Verify strict return type enforcement."""
        # 1. Path -> Dict
        mock_step_path = MagicMock()
        mock_step_path.func.return_value = Path("/tmp/foo.json")
        mock_step_path.name = "path_step"
        mock_step_path.artifact_paths = lambda x: [x["artifacts"][0]["path"]]
        self.runner.steps["path_step"] = mock_step_path

        m = self.runner._load_manifest()
        self.runner._execute_step(m, mock_step_path)  # ERROR FIX: Pass manifest + StepDef

        # Check manifest
        m = self.runner._load_manifest()
        art = m["steps"]["path_step"]["artifacts"]
        self.assertEqual(len(art), 1)
        self.assertEqual(art[0]["path"], "/tmp/foo.json")

        # 2. List[Path] -> Dict
        mock_step_list = MagicMock()
        mock_step_list.func.return_value = [Path("/tmp/a.json"), "/tmp/b.json"]
        mock_step_list.name = "list_step"
        mock_step_list.artifact_paths = lambda x: [a["path"] for a in x["artifacts"]]
        self.runner.steps["list_step"] = mock_step_list

        m = self.runner._load_manifest()
        self.runner._execute_step(m, mock_step_list)  # ERROR FIX: Pass manifest + StepDef

        m = self.runner._load_manifest()
        art = m["steps"]["list_step"]["artifacts"]
        self.assertEqual(len(art), 2)
        self.assertEqual(art[0]["path"], "/tmp/a.json")

        # 3. None -> TypeError (Caught and recorded as failure)
        mock_step_none = MagicMock()
        mock_step_none.func.return_value = None
        mock_step_none.name = "none_step"
        self.runner.steps["none_step"] = mock_step_none

        m = self.runner._load_manifest()
        self.runner._execute_step(m, mock_step_none)

        # Verify failure in manifest
        m = self.runner._load_manifest()
        step_data = m["steps"]["none_step"]
        self.assertEqual(step_data["status"], "FAILED")
        self.assertEqual(step_data["error"]["type"], "TypeError")
        self.assertIn("returned None", step_data["error"]["message"])

    def test_artifact_registry_and_lookup(self):
        """Verify global registry and deterministic lookup."""
        # Setup manifest with registry
        # Run step that produces typed artifact
        mock_step = MagicMock()
        mock_step.func.return_value = {"artifacts": [{"path": "out.json", "type": "result"}]}
        mock_step.name = "producer"
        mock_step.artifact_paths = lambda x: ["out.json"]
        self.runner.steps["producer"] = mock_step

        # Ensure file exists for mtime check - ERROR FIX: make sure parent exists
        self.runner.session_dir.mkdir(parents=True, exist_ok=True)
        (self.runner.session_dir / "out.json").touch()

        m = self.runner._load_manifest()
        self.runner._execute_step(m, mock_step)  # ERROR FIX: Pass manifest + StepDef

        # Verify Registry
        self.assertIn("artifacts_by_type", m)
        self.assertIn("result", m["artifacts_by_type"])
        self.assertEqual(
            min([x["path"] for x in m["artifacts_by_type"]["result"]]), "out.json"
        )  # Use min/list sort safety

        # Verify Lookup
        found = self.runner.get_artifact("result")
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "out.json")

    def test_stale_run_robustness(self):
        """Verify stale runs return full contract + debug fields."""
        # Create stale run
        run_id = "stale_run_v2"
        run_dir = self.runs_root / "sessions/hash" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "RUNNING",
                    "updated_at": "2020-01-01T00:00:00Z",  # Very stale
                    "current_step": "alignment",
                    "steps": {
                        "ingest": {"status": "COMPLETED"},
                        "asr": {"status": "COMPLETED"},
                        "alignment": {"status": "RUNNING"},
                    },
                }
            )
        )

        # Refresh index
        from server.services.runs_index import get_index

        get_index().refresh()

        # Test API logic (importing function to test without full server if possible, or mock)
        with patch("server.services.runs_index.RunsIndex.get_run") as mock_get:
            mock_get.return_value = {
                "run_id": run_id,
                "status": "RUNNING",
                "manifest_path": str(run_dir / "manifest.json"),
                "steps_completed": [],
            }

            from server.api.runs import get_run_status

            status = get_run_status(run_id)

            self.assertEqual(status["status"], "STALE")
            self.assertEqual(status["meta"]["snapshot_source"], "manifest")
            self.assertIsNotNone(status["meta"]["snapshot_reason"])
            self.assertGreater(status["meta"]["manifest_mtime"], 0)
            # Authoritative list check
            self.assertIn("ingest", status["steps_completed"])


if __name__ == "__main__":
    unittest.main()
