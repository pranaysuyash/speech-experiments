
import os
import sys
import pytest
import shutil
import subprocess
import zipfile
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path.cwd()))

from harness.media_ingest import sha256_file

# Gate for real execution (still uses same fixture mechanism)
RUN_REAL_E2E = os.environ.get("RUN_REAL_E2E") == "1"

@pytest.fixture
def smoke_mp4_path(tmp_path):
    """Generate a valid short MP4 for testing."""
    # We reuse the generation logic from test_real_smoke or just subprocess ffmpeg
    mp4_path = tmp_path / "smoke.mp4"
    # Generate 1s video with sine wave audio
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=128x72:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p",
        str(mp4_path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return mp4_path

@pytest.mark.skipif(not RUN_REAL_E2E, reason="Requires real environment components (ffmpeg, dependencies)")
def test_partial_bundle_on_failure(tmp_path: Path, smoke_mp4_path: Path):
    """
    Test Bundle Correctness Under Failure.
    
    Scenario:
      1. Run ingest (success) -> asr (fail).
      2. Verify exit code != 0.
      3. Verify manifest status == FAILED.
      4. Verify partial bundle zip exists.
      5. Verify zip contains ingest artifacts + manifest + log.
    """
    out_dir = tmp_path / "runs"
    script_path = Path("scripts/run_session.py").resolve()
    
    # We choose 'asr' as failure step so 'ingest' completes.
    fail_step = "asr" # using 'asr' because it follows ingest immediately
    
    # We use minimal args to reach ASR
    cmd = [
        sys.executable, str(script_path),
        "--input", str(smoke_mp4_path),
        "--out-dir", str(out_dir),
        "--steps", "ingest", "asr",
        "--pre",
        "--asr-size", "tiny", # Use tiny if it runs
    ]
    
    env = os.environ.copy()
    env["SESSION_FAIL_STEP"] = fail_step
    
    print(f"\n[Run] Executing with failure injection on {fail_step}...")
    
    # Expect failure
    try:
        subprocess.run(cmd, check=True, cwd=os.getcwd(), env=env, capture_output=True, text=True)
        pytest.fail("Command should have failed but succeeded")
    except subprocess.CalledProcessError as e:
        print(f"Command failed as expected with code {e.returncode}")
        print(f"STDOUT:\n{e.stdout}") # Debug
        print(f"STDERR:\n{e.stderr}")
        assert e.returncode != 0
        
    # Find session dir
    sessions_dir = out_dir / "sessions"
    assert sessions_dir.exists()
    # Should be one hash dir
    hash_dirs = list(sessions_dir.iterdir())
    assert len(hash_dirs) == 1
    run_dirs = list(hash_dirs[0].iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    
    # 1. Verify Manifest on disk
    manifest_path = run_dir / "manifest.json"
    assert manifest_path.exists()
    
    with open(manifest_path) as f:
        m = json.load(f)
        
    assert m["status"] == "FAILED"
    assert m["steps"]["ingest"]["status"] == "COMPLETED"
    assert m["steps"]["asr"]["status"] == "FAILED"
    # Check error message
    assert "Simulated failure" in str(m["steps"]["asr"]["error"])
    
    # 2. Verify Partial Bundle
    bundle_dir = run_dir / "bundle"
    assert bundle_dir.exists()
    bundle_files = list(bundle_dir.glob("*.zip"))
    assert len(bundle_files) == 1
    bundle_zip = bundle_files[0]
    
    print(f"Found partial bundle: {bundle_zip}")
    
    with zipfile.ZipFile(bundle_zip, 'r') as z:
        namelist = z.namelist()
        print(f"Bundle contents: {namelist}")
        
        # Must contain manifest
        assert "manifest.json" in namelist
        
        # Must contain run.log
        assert "logs/run.log" in namelist
        
        # Must contain ingest artifacts
        # We know ingest produces specific files, check partial path matching
        # Artifacts in zip are flat-ish? 
        # Check session.py _export_partial_bundle logic. 
        # Usually it preserves structure relative to run_dir? 
        # Or relative to 'artifacts' dir?
        # Let's check logic:
        # z.write(artifact_abs_path, arcname=f"artifacts/{task}/{artifact_name}")
        
        # Ingest produces processed_audio.wav
        # We expect artifacts/ingest/processed_audio.wav
        matches = [n for n in namelist if "processed_audio.wav" in n]
        assert len(matches) > 0, "processed_audio.wav missing from bundle"
        assert matches[0].startswith("artifacts/ingest/"), f"Unexpected path {matches[0]}"
        
        # ASR artifact should NOT be there (failed)
        asr_matches = [n for n in namelist if "asr_" in n and ".json" in n]
        assert len(asr_matches) == 0, "ASR artifact found despite failure"
