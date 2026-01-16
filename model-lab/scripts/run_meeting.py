#!/usr/bin/env python3
"""
Meeting recording workflow wrapper.

Usage:
    python scripts/run_meeting.py --input recording.wav --name "team-standup"
    python scripts/run_meeting.py --input video.mp4 --name "customer-call" --pre
"""

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
import shutil
import subprocess

# Regex to extract JSON sentinel line from stdout
_RESULT_RE = re.compile(r"^RUN_SESSION_RESULT=(\{.*\})\s*$", re.MULTILINE)


def _run_session_and_capture(cmd: list) -> Dict[str, Any]:
    """Run run_session.py and extract machine-parsable result.
    
    Streams all logs to console for debugging but extracts the 
    RUN_SESSION_RESULT sentinel line for deterministic output.
    
    Args:
        cmd: Command list to execute
        
    Returns:
        Dict with run_id, run_dir, console_url
        
    Raises:
        SystemExit: if command fails
        RuntimeError: if sentinel line not found
    """
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    # Always stream logs for debugging, even on success
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    # Extract JSON sentinel
    m = _RESULT_RE.search(proc.stdout or "")
    if not m:
        raise RuntimeError("run_session.py did not emit RUN_SESSION_RESULT=... line")

    return json.loads(m.group(1))


def main():
    parser = argparse.ArgumentParser(description="Run a meeting recording through the pipeline")
    parser.add_argument("--input", required=True, help="Input audio/video file")
    parser.add_argument("--name", required=True, help="Meeting name (no spaces, use dashes/underscores)")
    parser.add_argument("--pre", action="store_true", help="Enable preprocessing (normalization + silence removal)")
    parser.add_argument("--steps", help="Comma-separated steps to run (e.g., asr,chapters)")
    parser.add_argument("--force", action="store_true", help="Force re-run even if completed")
    parser.add_argument("--console-port", default="5174", help="Frontend console port (default: 5174)")
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    # Sanitize name
    safe_name = args.name.replace(" ", "-").replace("/", "-")
    
    # Create predictable folder structure
    # inputs/meetings/YYYY-MM/<name>_<timestamp>.<ext>
    now = datetime.now()
    year_month = now.strftime("%Y-%m")
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    ext = input_path.suffix
    
    meetings_dir = Path("inputs/meetings") / year_month
    meetings_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = f"{safe_name}_{timestamp}{ext}"
    output_path = meetings_dir / output_filename
    
    # Copy input to predictable location
    print(f"📁 Copying input to: {output_path}")
    shutil.copy2(input_path, output_path)
    
    # Build run_session.py command
    cmd = [
        sys.executable,
        "scripts/run_session.py",
        "--input", str(output_path),
        "--out-dir", "runs",
        "--console-port", args.console_port,
    ]
    
    if args.pre:
        cmd.append("--pre")
    
    if args.steps:
        cmd.extend(["--steps", args.steps])
    
    if args.force:
        cmd.append("--force")
    
    # Execute pipeline with robust capture
    print(f"🚀 Running pipeline: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = _run_session_and_capture(cmd)
    except (SystemExit, RuntimeError) as e:
        print(f"\n❌ Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1 if isinstance(e, RuntimeError) else e.code)
    
    # Print deterministic summary
    print("\n" + "=" * 60)
    print("✅ Meeting processed successfully!")
    print("=" * 60)
    print(f"📂 Input saved to: {output_path}")
    print(f"📂 Run Directory: {result['run_dir']}")
    print(f"🆔 Run ID: {result['run_id']}")
    print(f"🔗 Console: {result['console_url']}")

if __name__ == "__main__":
    main()
