
import threading
import time
import json
from pathlib import Path
from harness.session import SessionRunner

def mock_asr_progress(ctx):
    # Simulate work with progress updates
    for _ in range(15):
        time.sleep(1)
        if ctx.on_progress:
            ctx.on_progress()
    return {"status": "COMPLETED", "artifacts": []}

def test_semantic_progress():
    print("Setting up semantic progress test...")
    
    # Setup dummy session
    input_file = Path("traversal.wav") # Reuse existing
    if not input_file.exists():
        input_file.write_bytes(b"0" * 1024)
        
    runner = SessionRunner(
        input_path=str(input_file),
        output_dir="runs/test_progress",
        steps=["mock_step"]
    )
    
    # Inject mock step
    from harness.session import StepDef
    runner.steps["mock_step"] = StepDef(
        name="mock_step",
        deps=[],
        func=mock_asr_progress,
        artifact_paths=lambda x: []
    )
    
    runner._init_dirs()
    manifest_path = runner.manifest_path
    
    # Run in thread
    t = threading.Thread(target=runner._execute_step, args=(runner._default_manifest(), runner.steps["mock_step"]))
    t.start()
    
    print(f"Monitoring manifest: {manifest_path}")
    
    # Poll manifest for updates
    last_ts = None
    updates_seen = 0
    
    for _ in range(20): # 20 seconds max
        time.sleep(1)
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text())
            step_data = data.get("steps", {}).get("mock_step", {})
            ts = step_data.get("last_semantic_progress_at")
            
            if ts:
                print(f"Read timestamp: {ts}")
                if ts != last_ts:
                    updates_seen += 1
                    last_ts = ts
                    if updates_seen >= 2:
                        print("SUCCESS: Multiple semantic progress updates detected.")
                        return
            else:
                print("Waiting for timestamp...")
    
    print("FAILURE: Did not see multiple semantic progress updates.")

if __name__ == "__main__":
    test_semantic_progress()
