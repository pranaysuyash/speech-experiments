import json
import tempfile
from pathlib import Path

import pytest


class DummyPopen:
    def __init__(self, *args, **kwargs):
        self.pid = 12345

    def wait(self):
        return 0


class DummyRunner:
    def __init__(self, session_dir: Path, input_path: Path, run_id: str = "run123"):
        self.session_dir = session_dir
        self.input_path = input_path
        self.run_id = run_id
        self.manifest_path = self.session_dir / "manifest.json"


def test_launch_run_worker_writes_provenance_manifest(monkeypatch):
    from server.services import lifecycle

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        session_dir = tmp_path / "runs" / "sessions" / "abc" / "run123"
        session_dir.mkdir(parents=True, exist_ok=True)

        runner = DummyRunner(session_dir=session_dir, input_path=tmp_path / "inputs" / "input.wav")
        runner.input_path.parent.mkdir(parents=True, exist_ok=True)
        runner.input_path.write_bytes(b"dummy")

        run_request_data = {
            "schema_version": "1",
            "requested_at": "2026-01-27T12:00:00Z",
            "source": "workbench",
            "use_case_id": "uc123",
            "steps_preset": "full",
            "steps_requested": ["ingest", "asr"],
            "steps_custom": ["ingest", "asr", "diarization"],
            "pipeline_template": "full_meeting",
            "preprocessing": ["trim_silence"],
            "pipeline_config": {"name": "full_meeting", "steps": ["ingest", "asr"]},
            "config": {"device_preference": ["cpu"]},
            "filename_original": "input.wav",
            "bytes_uploaded": 5,
            "content_type": "audio/wav",
            "sha256": "deadbeef",
        }

        monkeypatch.setattr(lifecycle.subprocess, "Popen", DummyPopen)

        lifecycle.launch_run_worker(runner, run_request_data, background=False)

        manifest_path = runner.manifest_path
        assert manifest_path.exists(), "manifest.json should be created"

        manifest = json.loads(manifest_path.read_text())

        assert manifest.get("schema_version") == "1"
        assert manifest.get("run_id") == runner.run_id
        assert manifest.get("source") == "workbench"
        assert manifest.get("use_case_id") == "uc123"
        assert manifest.get("requested_at") == "2026-01-27T12:00:00Z"

        pipeline = manifest.get("pipeline", {})
        assert pipeline.get("steps_preset") == "full"
        assert pipeline.get("steps_requested") == ["ingest", "asr"]
        assert pipeline.get("steps_custom") == ["ingest", "asr", "diarization"]
        assert pipeline.get("pipeline_template") == "full_meeting"
        assert pipeline.get("preprocessing") == ["trim_silence"]
        assert pipeline.get("pipeline_config") == {"name": "full_meeting", "steps": ["ingest", "asr"]}
        assert pipeline.get("config_overrides") == {"device_preference": ["cpu"]}

        input_meta = manifest.get("input_metadata", {})
        assert input_meta.get("filename") == "input.wav"
        assert input_meta.get("size_bytes") == 5
        assert input_meta.get("content_type") == "audio/wav"
        assert input_meta.get("sha256") == "deadbeef"