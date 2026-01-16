"""
End-to-End Integration Test for Export Bundle.

Verifies that scripts/export_bundle.py correctly:
1. Discovers artifacts by hash
2. Generates a valid ZIP with manifest
3. Handles missing files gracefully
"""

import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
import hashlib
from pathlib import Path
import pytest

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.nlp_schema import compute_file_hash

class TestExportBundleE2E:
    
    @pytest.fixture
    def workspace(self):
        """Create a temporary workspace with standard directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "runs/asr").mkdir(parents=True)
            (root / "runs/diarization").mkdir(parents=True)
            (root / "runs/alignment").mkdir(parents=True)
            yield root

    def test_export_bundle_happy_path(self, workspace):
        """Verify full export flow with ASR, Diarization, and Alignment."""
        
        # 1. Setup Input
        input_file = workspace / "test_input.mp4"
        input_data = "mock_video_data_12345" # > 2MB if we wanted real file logic, but hash works on content
        input_file.write_text(input_data)
        
        # Since logic normally hashes first 2MB, we'll just mock the hash expectation
        # BUT export_bundle calls compute_input_hash which reads the file.
        # So we must ensure the artifacts match the ACTUAL hash of this file.
        real_input_hash = hashlib.sha256(input_data.encode()).hexdigest()
        
        # 2. Create Artifacts
        
        # ASR
        asr_data = {
            "inputs": {"audio_hash": real_input_hash},
            "output": {"text": "Hello world"}
        }
        asr_file = workspace / "runs/asr/mock_asr.json"
        asr_file.write_text(json.dumps(asr_data))
        
        # Compute hash using EXACTLY the method the script uses
        asr_hash = compute_file_hash(asr_file)
        
        # Diarization
        diar_data = {
            "inputs": {"audio_hash": real_input_hash},
            "output": {"segments": []}
        }
        diar_file = workspace / "runs/diarization/mock_diar.json"
        diar_file.write_text(json.dumps(diar_data))
        
        # Alignment (links to ASR)
        align_data = {
            "inputs": {
                "parent_artifact_hash": asr_hash,
                "parent_artifact_path": str(asr_file)
            },
            "output": {"segments": []}
        }
        align_file = workspace / "runs/alignment/mock_align.json"
        align_file.write_text(json.dumps(align_data))
        align_hash = compute_file_hash(align_file)

        
        # 3. Run Export Bundle
        # We need to run it as a subprocess to test the CLI
        # AND we need to point it to our temp workspace.
        # But the script uses RELATIVE paths (Path("runs")).
        # So we must set cwd to the workspace.
        
        # Copy script to workspace? No, script imports modules from repo.
        # Better: symlink repo modules into workspace or set PYTHONPATH.
        # Actually easier: Run from REPO_ROOT but patch the script's 'runs' dir?
        # The script defines RUNS_DIR = Path("runs").
        # We can patch it in a wrapper or just run this test logic BY IMPORTING the function
        # instead of subprocess. 
        # The user asked for: "runs scripts/export_bundle.py (subprocess) in non-dry mode"
        # This implies checking the CLI entry point.
        # To make "runs" resolvable, we can symlink "runs" in the CWD (workspace)
        # assuming the script uses CWD.
        
        # Force script to run in workspace
        cmd = [
             sys.executable, 
             str(REPO_ROOT / "scripts/export_bundle.py"),
             "--input", str(input_file),
             "--output", str(workspace / "bundle.zip")
        ]
        
        # We need the script to see 'runs' inside workspace.
        # The script does `Path("runs")`.
        result = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True)
        
        assert result.returncode == 0, f"Export failed: {result.stderr}"
        
        # 4. Verify Zip
        bundle_path = workspace / "bundle.zip"
        assert bundle_path.exists()
        
        with zipfile.ZipFile(bundle_path) as zf:
            names = zf.namelist()
            assert "manifest.json" in names
            assert "artifacts/asr_mock_asr.json" in names
            assert "artifacts/diarization_mock_diar.json" in names
            assert "artifacts/alignment_mock_align.json" in names
            
            # Verify Manifest
            manifest = json.loads(zf.read("manifest.json"))
            assert manifest["input_hash"] == real_input_hash
            assert manifest["artifacts"]["asr"]["file"] == "artifacts/asr_mock_asr.json"
            assert manifest["artifacts"]["alignment"]["hash"] == align_hash

    def test_export_bundle_no_artifacts(self, workspace):
        """Verify behavior when no artifacts exist."""
        input_file = workspace / "empty.mp4"
        input_file.write_text("void")
        
        cmd = [
             sys.executable, 
             str(REPO_ROOT / "scripts/export_bundle.py"),
             "--input", str(input_file),
             "--output", str(workspace / "empty.zip")
        ]
        
        result = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True)
        
        # Should fail with exit code 1
        assert result.returncode == 1
        assert "No artifacts found" in result.stderr

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
