"""
End-to-End Integration Test for Chapters.

Verifies:
1. model_app.py triggers run_chapters.py
2. run_chapters.py produces chapters.json with correct config
3. export_bundle.py includes chapters in zip
"""

import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.nlp_schema import compute_file_hash


class TestChaptersE2E:
    @pytest.fixture
    def workspace(self):
        """Create a temporary workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "runs/asr").mkdir(parents=True)
            (root / "runs/alignment").mkdir(parents=True)
            (root / "runs/nlp/chapters").mkdir(parents=True)
            yield root

    def test_chapters_flow(self, workspace):
        """Test chapters extraction via CLI and bundling."""

        # 1. Setup Data - Correct Chain order for Export integrity
        # Audio
        audio_file = workspace / "meeting.mp4"
        audio_file.write_text("mock_audio_content")
        audio_hash = hashlib.sha256(audio_file.read_bytes()).hexdigest()

        # ASR
        asr_file = workspace / "runs/asr/mock_asr.json"
        asr_data = {"inputs": {"audio_hash": audio_hash, "media_path": str(audio_file)}}
        asr_file.write_text(json.dumps(asr_data))
        asr_hash = compute_file_hash(asr_file)

        # Alignment
        align_file = workspace / "runs/alignment/alignment_mock.json"
        align_data = {
            "segments": [
                {"start_s": 0.0, "end_s": 5.0, "text": "Welcome everyone.", "speaker_id": "spk1"},
                {"start_s": 5.0, "end_s": 10.0, "text": "Thanks for coming.", "speaker_id": "spk2"},
            ],
            "metrics": {
                "total_duration_s": 10.0,
                "coverage_ratio": 1.0,
                "unknown_ratio": 0.0,
                "speaker_switch_count": 1,
                "speaker_distribution": {},
                "assigned_duration_s": 10.0,
            },
            "source_asr_path": str(asr_file),
            "source_diarization_path": "mock",
            "inputs": {"parent_artifact_hash": asr_hash, "parent_artifact_path": str(asr_file)},
        }
        align_file.write_text(json.dumps(align_data))

        # 2. Run run_chapters.py
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts/run_chapters.py"),
            "--from-artifact",
            str(align_file),
            "--embedding-cache-dir",
            str(workspace / "cache"),
        ]

        # Run in workspace
        result = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True)
        assert result.returncode == 0, f"Run failed: {result.stderr}"

        # Parse output for artifact path
        artifact_path = None
        for line in result.stdout.splitlines():
            if line.startswith("ARTIFACT_PATH:"):
                artifact_path = line.split(":", 1)[1].strip()
                break

        assert artifact_path, f"No ARTIFACT_PATH printed. Stderr: {result.stderr}"
        assert Path(workspace / artifact_path).exists()

        # Verify Content
        with open(workspace / artifact_path) as f:
            data = json.load(f)
            assert "config" in data
            assert data["config"]["model"] == "all-MiniLM-L6-v2"
            assert len(data["chapters"]) > 0

        # 3. Export Bundle
        export_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts/export_bundle.py"),
            "--input",
            str(audio_file),
            "--output",
            str(workspace / "bundle.zip"),
        ]

        res_exp = subprocess.run(export_cmd, cwd=workspace, capture_output=True, text=True)
        assert res_exp.returncode == 0, f"Export failed: {res_exp.stderr}"

        # Verify Zip
        with zipfile.ZipFile(workspace / "bundle.zip") as zf:
            names = zf.namelist()
            # Name format: artifacts/TASK_filename
            # Task: nlp/chapters -> nlp_chapters
            # File: chapters_alignment_mock.json
            # Result: artifacts/nlp_chapters_chapters_alignment_mock.json
            expected_name = f"artifacts/nlp_chapters_chapters_{align_file.stem}.json"

            # Verify Manifest
            manifest = json.loads(zf.read("manifest.json"))

            if expected_name not in names:
                pytest.fail(
                    f"Chapters artifact {expected_name} not found in zip.\nFiles: {names}\nSTDOUT:\n{res_exp.stdout}\nSTDERR:\n{res_exp.stderr}"
                )

            assert "nlp/chapters" in manifest["artifacts"]
            assert manifest["schema_version"] == "1.1.0"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
