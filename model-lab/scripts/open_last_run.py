#!/usr/bin/env python3
"""
Open the most recently created run in the console.

Usage:
    python scripts/open_last_run.py
"""

import json
from pathlib import Path


def main():
    runs_root = Path("runs/sessions")

    if not runs_root.exists():
        print("No runs found. Run a session first!")
        return

    # Find all manifests
    manifests = list(runs_root.glob("*/*/manifest.json"))

    if not manifests:
        print("No completed runs found.")
        return

    # Sort by started_at from manifest
    runs = []
    for manifest_path in manifests:
        try:
            data = json.loads(manifest_path.read_text())
            run_id = data.get("run_id", manifest_path.parent.name)
            started_at = data.get("started_at", "")
            runs.append((run_id, started_at, manifest_path))
        except Exception:
            continue

    if not runs:
        print("No valid runs found.")
        return

    # Sort by started_at descending
    runs.sort(key=lambda x: x[1], reverse=True)

    latest_run_id = runs[0][0]
    latest_started = runs[0][1]

    console_url = f"http://localhost:5174/#/runs/{latest_run_id}"

    print(f"Latest run: {latest_run_id}")
    print(f"Started at: {latest_started}")
    print(f"\n🔗 Console URL:\n{console_url}")
    print("\nNote: Make sure the frontend dev server is running on port 5174.")


if __name__ == "__main__":
    main()
