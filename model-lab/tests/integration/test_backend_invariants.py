
import pytest
import time
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

from harness.session import SessionRunner
from harness.asr import ResolvedASRConfig
from server.services.lifecycle import kill_run

class TestBackendInvariants:
    
    @pytest.fixture
    def session_dir(self):
        d = Path(tempfile.mkdtemp())
        # Ensure log dir exists for kill logic
        (d / "worker.log").write_text("")
        yield d
        shutil.rmtree(d)

    @pytest.fixture
    def mock_manifest(self, session_dir):
        m = {
            "run_id": "test_run",
            "status": "RUNNING",
            "steps": {},
            "config": {},
            "created_at": "2024-01-01T00:00:00Z",
            "worker_pid": 12345
        }
        (session_dir / "manifest.json").write_text(json.dumps(m))
        return m

    def test_kill_run_idempotence(self, session_dir, mock_manifest):
        """Verify kill_run returns correct outcomes for various states."""
        
        # 1. Non-existent run
        with patch("server.services.runs_index.get_index") as mock_get_index:
            mock_index = MagicMock()
            mock_get_index.return_value = mock_index
            mock_index.get_run.return_value = None
            
            success, outcome = kill_run("non_existent_id")
            assert not success
            assert outcome == "not_found"

        # 2. Run exists, no PID file (Simulate forced cancel)
        with patch("server.services.runs_index.get_index") as mock_get_index:
            mock_index = MagicMock()
            mock_get_index.return_value = mock_index
            mock_index.get_run.return_value = {
                "run_id": "test_run",
                "status": "RUNNING",
                "manifest_path": str(session_dir / "manifest.json")
            }
            
            # Ensure PID file NOT present
            pid_file = session_dir / "worker.pid"
            if pid_file.exists(): pid_file.unlink()

            # Ensure Manifest has NO worker_pid for this test case to trigger forced_cancel
            m_data = json.loads((session_dir / "manifest.json").read_text())
            if "worker_pid" in m_data:
                del m_data["worker_pid"]
            (session_dir / "manifest.json").write_text(json.dumps(m_data))

            success, outcome = kill_run("test_run")
            assert success
            assert outcome == "forced_cancel"
            
            # Verify status updated to CANCELLED
            m = json.loads((session_dir / "manifest.json").read_text())
            assert m["status"] == "CANCELLED"

        # 3. Run exists, PID file exists but process dead
        (session_dir / "worker.pid").write_text("999999") 
        mock_manifest["status"] = "RUNNING"
        (session_dir / "manifest.json").write_text(json.dumps(mock_manifest))
        
        with patch("server.services.runs_index.get_index") as mock_get_index, \
             patch("os.kill", side_effect=ProcessLookupError):
            mock_index = MagicMock()
            mock_get_index.return_value = mock_index
            mock_index.get_run.return_value = {
                "run_id": "test_run",
                "status": "RUNNING",
                "manifest_path": str(session_dir / "manifest.json")
            }
            
            success, outcome = kill_run("test_run")
            assert success
            assert outcome == "already_dead"
            
            m = json.loads((session_dir / "manifest.json").read_text())
            assert m["status"] == "CANCELLED"

    def test_kill_already_terminal(self, session_dir, mock_manifest):
        # 4. Run already FAILED
        mock_manifest["status"] = "FAILED"
        (session_dir / "manifest.json").write_text(json.dumps(mock_manifest))
        
        with patch("server.services.runs_index.get_index") as mock_get_index:
             mock_index = MagicMock()
             mock_get_index.return_value = mock_index
             mock_index.get_run.return_value = {
                 "run_id": "test_run",
                 "status": "FAILED",
                 "manifest_path": str(session_dir / "manifest.json")
             }
             
             success, outcome = kill_run("test_run")
             assert success
             assert outcome == "already_terminal"
             
             m = json.loads((session_dir / "manifest.json").read_text())
             assert m["status"] == "FAILED"

    def test_asr_config_persistence(self, session_dir):
        """Verify requested and resolved configs are promoted to manifest."""
        
        input_path = session_dir / "input.wav"
        input_path.touch()
        
        user_config = {
            "asr": {
                "model_type": "faster_whisper", 
                "device": "mps",
                "model_name": "default"
            }
        }
        
        # FIXED: Removed session_id arg
        runner = SessionRunner(
            input_path=input_path,
            output_dir=session_dir.parent,
            config=user_config
        )
        
        # Initialize run to create manifest
        runner._init_dirs()
        m = runner._load_manifest()
        m["status"] = "RUNNING"
        runner._save_manifest(m)
        
        from harness.asr import ResolvedASRConfig
        resolved = ResolvedASRConfig(
            model_id="faster_whisper:large-v3",
            source="hf",
            device="cpu", # Fallback expected
            reason="mps_unsupported_by_backend",
            language="auto"
        )
        
        step_result = {
            "result": {},
            "resolved_config": resolved.to_dict(),
            "requested_config": user_config["asr"]
        }
        
        mock_step_def = MagicMock()
        mock_step_def.name = "asr"
        mock_step_def.artifact_paths.return_value = []
        mock_step_def.func.return_value = step_result
        
        runner.steps["asr"] = mock_step_def
        
        try:
            runner._execute_step(m, mock_step_def)
        except Exception:
            pass
            
        m_final = json.loads(runner.manifest_path.read_text())
        step_entry = m_final["steps"]["asr"]
        
        assert "resolved_config" in step_entry
        assert step_entry["resolved_config"]["device"] == "cpu"
        assert step_entry["resolved_config"]["reason"] == "mps_unsupported_by_backend"
        
        assert "requested_config" in step_entry
        assert step_entry["requested_config"]["device"] == "mps"

    def test_failure_propagation(self, session_dir):
        """Verify failure in a step propagates to top-level status."""
        
        input_path = session_dir / "input.wav"
        input_path.touch()
        
        # FIXED: constructor signature
        runner = SessionRunner(
            input_path=input_path, 
            output_dir=session_dir.parent
        )
        runner._init_dirs()
        
        def failing_func(ctx):
             raise RuntimeError("Boom")
             
        step_def = MagicMock()
        step_def.name = "failing_step"
        step_def.func = failing_func
        step_def.deps = []
        # Mock artifact paths to return empty list or it might fail if func fails before returns?
        # Actually _execute_step calls func inside try/except.
        runner.steps["failing_step"] = step_def
        
        m = runner._load_manifest()
        m["status"] = "RUNNING"
        runner._save_manifest(m)
        
        # Run it
        try:
             runner._execute_step(m, step_def)
        except RuntimeError:
             pass 
             
        m_final = json.loads(runner.manifest_path.read_text())
        
        assert m_final["status"] == "FAILED"
        assert m_final["failure_step"] == "failing_step"
        assert m_final["steps"]["failing_step"]["status"] == "FAILED"
        assert "Boom" in m_final["error_message"]

    def test_status_regression_prevention(self, session_dir):
        """Verify that external CANCELLED status is not overwritten by COMPLETED."""
        input_path = session_dir / "input.wav"
        input_path.touch()
        
        runner = SessionRunner(
            input_path=input_path, 
            output_dir=session_dir.parent
        )
        runner._init_dirs()
        
        # Define a step that simulates external kill during execution
        def step_func(ctx):
             # Write CANCELLED to manifest on disk
             m = json.loads(runner.manifest_path.read_text())
             m["status"] = "CANCELLED"
             runner.manifest_path.write_text(json.dumps(m))
             return {}

        step_def = MagicMock()
        step_def.name = "step_1"
        step_def.func = step_func
        step_def.deps = []
        step_def.artifact_paths.return_value = []
        
        # Override _register_steps to prevent adding ingest/asr/bundle
        runner._register_steps = MagicMock()
        runner.steps["step_1"] = step_def
        
        # Mock ingest to avoid ffmpeg failure on empty file
        # harness.media_ingest is imported inside session.py usually, or we can patch where it's used
        with patch("harness.session.ingest_media") as mock_ingest:
            mock_ingest.return_value = {"duration_s": 1.0, "processed_audio_path": str(input_path), "audio_content_hash": "abc"}
            # Run
            runner.run()
        
        # Merge verification:
        m_final = json.loads(runner.manifest_path.read_text())
        
        # 1. Status monotonic: CANCELLED preserved
        assert m_final["status"] == "CANCELLED"
        
        # 2. Step Status: Downgraded to CANCELLED (not COMPLETED)
        assert m_final["steps"]["step_1"]["status"] == "CANCELLED"
        assert m_final["steps"]["step_1"]["error"]["type"] == "Cancelled"
        
        # 3. Merge Safety: External fields preserved?
        # Ideally the step_func wrote specific fields.
        # But wait, step_func writes manifest using json.dump(m). 
        # If step_func simulates external kill, it might write extra fields.
        # Let's verify that.
        # But wait, step_func implementation above:
        # m["status"] = "CANCELLED"
        # runner.manifest_path.write_text(json.dumps(m))
        # It didn't add extra fields. Let's add one.
        pass

    def test_status_regression_prevention_strict(self, session_dir):
        """Verify SAFE MERGE: External fields + Cancelled status vs Runner artifacts."""
        input_path = session_dir / "input.wav"
        input_path.touch()
        
        runner = SessionRunner(
            input_path=input_path, 
            output_dir=session_dir.parent
        )
        runner._init_dirs()
        
        def step_func(ctx):
             # Simulating concurrent external update (e.g. from kill_run)
             # Read current from disk (runner wrote RUNNING)
             d = json.loads(runner.manifest_path.read_text())
             d["status"] = "CANCELLED"
             d["error"] = {"type": "UserCancelled", "message": "Manual Kill"} # External field
             d["kill_meta"] = "preserved" # Random field
             d["terminated_by"] = "admin" # Another random field
             runner.manifest_path.write_text(json.dumps(d))
             
             # Return result as if work finished
             return {"some_result": 123}

        step_def = MagicMock()
        step_def.name = "concurrent_step"
        step_def.func = step_func
        step_def.deps = []
        step_def.artifact_paths.return_value = []
        
        runner._register_steps = MagicMock()
        runner.steps["concurrent_step"] = step_def
        
        with patch("harness.session.ingest_media") as mock_ingest:
            mock_ingest.return_value = {"duration_s": 1.0, "processed_audio_path": str(input_path), "audio_content_hash": "abc"}
            runner.run()
        
        m_final = json.loads(runner.manifest_path.read_text())
        
        # Assertions
        assert m_final["status"] == "CANCELLED" # Regressions prevention
        assert m_final["error"]["type"] == "UserCancelled" # Merge prevention (runner error didn't overwrite)
        
        # Verify strict arbitrary metadata preservation
        assert m_final.get("kill_meta") == "preserved"
        assert m_final.get("terminated_by") == "admin"
        
        # Verify step artifact/result is present despite cancellation
        step_entry = m_final["steps"]["concurrent_step"]
        assert step_entry["status"] == "CANCELLED" # Step status downgraded
        assert step_entry["result"]["some_result"] == 123 # Data preserved!
        assert step_entry["error"]["type"] == "Cancelled" # Step error reflects outcome
