import json
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from server.main import app

class TestDevicePropagation:
    
    def test_device_preference_is_persisted_in_manifest(self, tmp_path, monkeypatch):
        """
        Verify that device_preference passed to API is persisted in manifest.
        """
        # Override inputs/runs root for safety
        inputs_root = tmp_path / "inputs"
        runs_root = tmp_path / "runs"
        monkeypatch.setenv("MODEL_LAB_INPUTS_ROOT", str(inputs_root))
        monkeypatch.setenv("MODEL_LAB_RUNS_ROOT", str(runs_root))
        
        # Ensure roots exist
        inputs_root.mkdir(parents=True)
        runs_root.mkdir(parents=True)
        
        client = TestClient(app)
        
        # 1. API Contract: Send request with device_preference
        # Workbench API uses multipart/form-data
        files = {"file": ("test.wav", b"fake audio content", "audio/wav")}
        data = {
            "use_case_id": "uc_test",
            "steps_preset": "ingest", # minimal preset
            "config": json.dumps({
                "device_preference": ["mps", "cpu"],
                "some_misc_flag": True
            })
        }
        
        with monkeypatch.context() as m:
            # Mock worker launch to avoid actual process
            # Capture what was passed to verify API parsed/merged config correctly
            mock_launch = MagicMock(return_value={"worker_pid": 123})
            m.setattr("server.api.workbench.try_acquire_worker", lambda: True)
            m.setattr("server.api.workbench.launch_run_worker", mock_launch)
            
            resp = client.post("/api/workbench/runs", files=files, data=data)
            assert resp.status_code == 200, resp.text
            
            # Verify passed data
            assert mock_launch.call_count == 1
            # args[0] is runner, args[1] is run_request_data
            run_req = mock_launch.call_args[0][1]
            
            assert run_req["config"]["device_preference"] == ["mps", "cpu"]
            assert run_req["config"]["some_misc_flag"] is True

    def test_device_preference_reaches_asr_model_load(self, monkeypatch, tmp_path):
        """
        Verify that run_asr correctly resolves preference and calls load_model.
        """
        import harness.asr as asr
        
        # 1. Mock support checker to simluate 'mps' unsupported
        def fake_is_device_supported(*, backend, device):
            return False 
        
        # Mock ModelRegistry.load_model
        # Must return bundle with capabilities=["asr"]
        load_model_mock = MagicMock(return_value={
            "model": "fake_bundle", 
            "capabilities": ["asr"],
            "asr": {"transcribe": MagicMock(return_value={"segments": [], "info": MagicMock()})}
        })
        monkeypatch.setattr(asr.ModelRegistry, "load_model", load_model_mock)
        
        # Mock ingest_media to return fake paths
        monkeypatch.setattr(asr, "ingest_media", lambda i, o, c: {
            "processed_audio_path": str(i), 
            "audio_content_hash": "abc"
        })
        
        # Mock soundfile to avoid reading fake wav
        mock_sf = MagicMock()
        mock_sf.read.return_value = ([], 16000)
        monkeypatch.setattr("soundfile.read", mock_sf.read)
        
        # Setup Run Context
        session_dir = tmp_path / "session"
        session_dir.mkdir()
        input_path = session_dir / "input.wav"
        input_path.write_text("fake wav")
        output_dir = session_dir / "artifacts"
        
        config = {
            "model_type": "faster_whisper",
            "device_preference": ["mps", "cpu"],
            "pre_ops": [],
            "dataset_def": None
        }
        
        # Execute run_asr (the worker step)
        asr.run_asr(
            input_path=input_path,
            output_dir=output_dir,
            config=config, 
            progress_callback=None
        )
        
        # Assert load_model was called with resolved device "cpu"
        # Because we passed "faster_whisper" and "preference=['mps','cpu']"
        # harness/asr.py logic should resolve to 'cpu' because faster_whisper skips mps
        assert load_model_mock.call_count == 1
        _, kwargs = load_model_mock.call_args
        assert kwargs.get("device") == "cpu"
        
        # Test 2: Valid Preference
        load_model_mock.reset_mock()
        config2 = {
            "model_type": "distil_whisper",
            "device_preference": ["mps", "cpu"], 
        }
        asr.run_asr(input_path, output_dir, config2)
        _, kwargs = load_model_mock.call_args
        assert kwargs.get("device") == "mps"
