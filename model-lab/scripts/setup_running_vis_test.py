import json
import shutil
import time
from pathlib import Path

# Configuration
RUNS_DIR = Path("runs/sessions")
RUN_ID = f"vis_test_{int(time.time())}"
INPUT_HASH = "test_hash"
SESSION_DIR = RUNS_DIR / INPUT_HASH / RUN_ID
ARTIFACTS_DIR = SESSION_DIR / "artifacts"


def setup_test_run():
    print(f"Setting up test run: {RUN_ID}")

    # Clean up previous tests
    if SESSION_DIR.exists():
        shutil.rmtree(SESSION_DIR)

    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "ingest").mkdir(parents=True, exist_ok=True)
    (SESSION_DIR / "input").mkdir(parents=True, exist_ok=True)

    # Create dummy input file
    input_file = SESSION_DIR / "input/test_audio.wav"
    input_file.write_bytes(b"0" * 1024 * 1024 * 5)  # 5MB dummy file

    # Create dummy artifact
    (ARTIFACTS_DIR / "ingest/processed_audio.wav").write_text("dummy content")

    manifest = {
        "run_id": RUN_ID,
        "status": "RUNNING",
        "started_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() - 120)
        ),  # 2 mins ago
        "updated_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() - 40)
        ),  # 40s ago (Should trigger "Process may still be alive")
        "steps_completed": ["ingest"],
        "current_step": "asr",
        "input": {"original_path": str(input_file.absolute()), "input_hash": INPUT_HASH},
        "config": {
            "asr": {"model_name": "default", "device": "cuda"},  # TESTING DEFAULT
            "diarization": {"enabled": True},
            "preprocessing": {"normalize": True, "trim_silence": False},
        },
        "steps": {
            "ingest": {
                "status": "COMPLETED",
                "artifacts": [{"path": str(ARTIFACTS_DIR / "ingest/processed_audio.wav")}],
            },
            "asr": {"status": "RUNNING", "artifacts": []},
        },
    }

    (SESSION_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Force index refresh
    print("Run created. Please browse to:")
    print(f"http://localhost:5174/lab/runs/{RUN_ID}")


if __name__ == "__main__":
    setup_test_run()
