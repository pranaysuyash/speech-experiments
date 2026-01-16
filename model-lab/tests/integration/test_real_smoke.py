from __future__ import annotations

import os
import subprocess
import shutil
import pytest
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from harness.media_ingest import sha256_file

# Gate this test behind an environment variable
RUN_REAL_E2E = os.environ.get("RUN_REAL_E2E") == "1"

@pytest.fixture
def smoke_mp4_path(tmp_path: Path) -> Path:
    """Generates a 5s synthetic video with audio."""
    path = tmp_path / "smoke.mp4"
    # Generate 5s of color bars + 440Hz tone
    # -f lavfi -i testsrc=duration=5:size=1280x720:rate=30
    # -f lavfi -i sine=frequency=440:duration=5
    # -c:v libx264 -c:a aac
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
        "-c:v", "libx264", "-c:a", "aac",
        "-pix_fmt", "yuv420p", # Ensure compatible pixel format
        str(path)
    ]
    subprocess.run(cmd, check=True)
    return path

@pytest.mark.skipif(not RUN_REAL_E2E, reason="Real E2E tests not enabled (RUN_REAL_E2E=1 required)")
def test_real_smoke_ingest_asr(tmp_path: Path, smoke_mp4_path: Path) -> None:
    """
    Real E2E smoke test: Ingest -> ASR on a standardized MP4.
    Verifies:
      1. Pipeline runs via CLI
      2. Ingest produces wav and manifest results
      3. ASR runs and produces artifact
      4. Resume works (skipping steps on second run)
    """
    out_dir = tmp_path / "runs"
    script_path = Path("scripts/run_session.py").resolve()
    
    # --- Run 1: Execution ---
    cmd = [
        sys.executable, str(script_path),
        "--input", str(smoke_mp4_path),
        "--out-dir", str(out_dir),
        "--steps", "ingest", "asr",
        "--pre",
        "--asr-size", "tiny",
        "--compute-type", "int8",
    ]
    
    print(f"\n[Run 1] Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=os.getcwd(), capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[Run 1] FAILED with return code {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        raise e
    
    # Verification
    input_hash = sha256_file(smoke_mp4_path)
    # Finding the session dir is slightly tricky because run_id is timestamped.
    # But checking the output dir structure:
    # out_dir / "sessions" / input_hash / <run_id>
    session_root = out_dir / "sessions" / input_hash
    assert session_root.exists(), f"Session root not found: {session_root}"
    
    run_dirs = list(session_root.iterdir())
    assert len(run_dirs) == 1, f"Expected exactly 1 run dir, found: {run_dirs}"
    run_dir = run_dirs[0]
    
    manifest_path = run_dir / "manifest.json"
    assert manifest_path.exists()
    
    with open(manifest_path, 'r') as f:
        m = json.load(f)
        
    assert m["status"] == "COMPLETED"
    
    # verify Ingest
    ingest_step = m["steps"].get("ingest", {})
    assert ingest_step["status"] == "COMPLETED"
    ingest_res = ingest_step.get("result", {})
    assert ingest_res["source_media_hash"] == input_hash
    assert "preprocess_hash" in ingest_res
    
    # Verify processed audio exists
    processed_audio = Path(ingest_res["processed_audio_path"])
    assert processed_audio.exists()
    assert ingest_res["audio_content_hash"] == sha256_file(processed_audio)
    
    # Verify ASR
    asr_step = m["steps"].get("asr", {})
    assert asr_step["status"] == "COMPLETED"
    asr_artifacts = asr_step.get("artifacts", [])
    assert len(asr_artifacts) > 0
    asr_json_path = Path(asr_artifacts[0]["path"])
    assert asr_json_path.exists()
    assert asr_json_path.stat().st_size > 0
    
    # Capture state for Resume check
    t1_ingest_start = ingest_step["started_at"]
    t1_asr_start = asr_step["started_at"]
    
    # --- Run 2: Resume ---
    # We must explicitly point to the previous run using --resume-from
    # because default behavior generates a NEW timestamped run_id.
    
    cmd_resume = cmd + ["--resume-from", str(run_dir)]
    
    print(f"\n[Run 2] Executing Resume: {' '.join(cmd_resume)}")
    subprocess.run(cmd_resume, check=True, cwd=os.getcwd())
    
    # Re-verify Manifest
    with open(manifest_path, 'r') as f:
        m2 = json.load(f)
        
    assert m2["status"] == "COMPLETED"
    
    ingest_step_2 = m2["steps"]["ingest"]
    asr_step_2 = m2["steps"]["asr"]
    
    # Timestamps should match exactly if skipped
    assert ingest_step_2["started_at"] == t1_ingest_start, "Ingest should have been skipped (same started_at)"
    assert asr_step_2["started_at"] == t1_asr_start, "ASR should have been skipped (same started_at)"
    
    print("\n[SUCCESS] Real smoke test passed.")
