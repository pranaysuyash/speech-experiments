
import threading
import time
import json
import os
from pathlib import Path
from harness.session import SessionRunner, StepDef

def mock_step_long(ctx):
    # Simulate work with progress updates for 30s
    print("Mock step starting...")
    for i in range(30):
        time.sleep(1)
        if ctx.on_progress:
            ctx.on_progress()
            # print(f"Emitted progress {i}")
    print("Mock step done.")
    return {"status": "COMPLETED", "artifacts": []}

def run_server():
    # Assume server is running on port 8000 via dev.sh
    pass

def setup_run():
    print("Setting up live semantic progress run...")
    
    # Setup dummy session
    run_id = f"semantic_vis_{int(time.time())}"
    input_file = Path("traversal.wav")
    if not input_file.exists():
        input_file.write_bytes(b"0" * 1024)
        
    runner = SessionRunner(
        input_path=str(input_file),
        output_dir=f"runs/{run_id}",
        steps=["mock_long"]
    )
    
    # Inject mock step
    runner.steps["mock_long"] = StepDef(
        name="mock_long",
        deps=[],
        func=mock_step_long,
        artifact_paths=lambda x: []
    )
    
    runner._init_dirs()
    
    # Run in thread
    t = threading.Thread(target=runner._execute_step, args=(runner._default_manifest(), runner.steps["mock_long"]))
    t.start()
    
    print(f"\nRun created: {runner.run_id}")
    print(f"URL: http://localhost:5174/runs/{runner.run_id}")
    print("Waiting 20s for you to verify UI...")
    time.sleep(20)
    print("Test finished.")

if __name__ == "__main__":
    setup_run()
