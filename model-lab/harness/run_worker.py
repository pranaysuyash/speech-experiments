#!/usr/bin/env python3
"""
Run worker - executes a session in a subprocess.

This is the subprocess entrypoint called by the API server.
It runs completely independently of the web server process.

Usage:
    python -m harness.run_worker --run-dir /path/to/run/dir

The run directory must contain:
    - run_request.json: Run configuration
    - manifest.json: Current run state (will be updated)

All output goes to files in the run directory.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

# Set up logging to file in run directory
def setup_logging(run_dir: Path) -> None:
    log_file = run_dir / "worker.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stderr),
        ],
    )

def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON atomically via temp file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)

def main() -> int:
    parser = argparse.ArgumentParser(description="Run session worker")
    parser.add_argument("--run-dir", required=True, help="Path to run directory")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir).resolve()
    
    if not run_dir.exists():
        print(f"ERROR: Run directory does not exist: {run_dir}", file=sys.stderr)
        return 1
    
    setup_logging(run_dir)
    logger = logging.getLogger("run_worker")
    
    manifest_path = run_dir / "manifest.json"
    request_path = run_dir / "run_request.json"
    
    if not request_path.exists():
        logger.error(f"run_request.json not found in {run_dir}")
        return 1
    
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        logger.info(f"Starting run: {request.get('run_id', 'unknown')}")
        
        # Import SessionRunner here to avoid loading models at import time
        from harness.session import SessionRunner
        
        # Get the input path from the request
        input_path = Path(request.get("input_path", ""))
        if not input_path.exists():
            # Try relative to runs root
            runs_root = run_dir.parent.parent  # runs/sessions/hash/run_id -> runs
            input_path = runs_root / request.get("input_rel_path", "")
        
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            # Write FAILED to manifest
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            else:
                manifest = {"status": "RUNNING"}
            manifest["status"] = "FAILED"
            manifest["error"] = f"Input file not found: {input_path}"
            atomic_write_json(manifest_path, manifest)
            return 1
        
        # Get steps from request
        steps = request.get("steps")
        
        # Create runner - it will use the existing run directory
        # run_dir is base/sessions/hash/run_id
        # we need base
        runs_root = run_dir.parent.parent.parent
        runner = SessionRunner(
            input_path, 
            runs_root,
            steps=steps,
            config={
                **request.get("config", {}), # Ingest config from request
                "resume_from": str(run_dir)
            },
            resume=True
        )
        
        # Run the pipeline
        logger.info("Executing pipeline...")
        runner.run()
        logger.info("Pipeline completed successfully")
        
        return 0
        
    except Exception as e:
        logger.exception(f"Run failed with exception: {e}")
        
        # Write FAILED to manifest
        try:
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            else:
                manifest = {"status": "RUNNING"}
            manifest["status"] = "FAILED"
            manifest["error"] = str(e)
            manifest["traceback"] = traceback.format_exc()
            atomic_write_json(manifest_path, manifest)
        except Exception:
            pass
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
