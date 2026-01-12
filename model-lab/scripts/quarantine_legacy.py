#!/usr/bin/env python3
"""
One-time sweep to quarantine all legacy runs without provenance.

Usage:
    python scripts/quarantine_legacy.py --dry-run  # List what would be quarantined
    python scripts/quarantine_legacy.py            # Actually quarantine
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.quarantine_run import quarantine_run, load_manifest, save_manifest


def find_legacy_runs() -> list:
    """Find all runs missing provenance field."""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return []
    
    legacy = []
    for run_file in runs_dir.glob("**/*.json"):
        if run_file.name == "summary.json":
            continue
        
        try:
            with open(run_file) as f:
                data = json.load(f)
            
            if "provenance" not in data:
                legacy.append(run_file)
        except json.JSONDecodeError:
            legacy.append(run_file)  # Invalid JSON = quarantine
    
    return legacy


def main():
    dry_run = "--dry-run" in sys.argv
    
    legacy_runs = find_legacy_runs()
    
    if not legacy_runs:
        print("✅ No legacy runs found - all runs have provenance!")
        return 0
    
    print(f"Found {len(legacy_runs)} runs without provenance:")
    for run in legacy_runs[:10]:
        print(f"  - {run}")
    if len(legacy_runs) > 10:
        print(f"  ... and {len(legacy_runs) - 10} more")
    
    if dry_run:
        print("\n[DRY RUN] Would quarantine the above runs.")
        print("Run without --dry-run to actually quarantine.")
        return 0
    
    print(f"\nQuarantining {len(legacy_runs)} legacy runs...")
    
    success = 0
    for run_file in legacy_runs:
        if quarantine_run(run_file, "legacy_no_provenance"):
            success += 1
    
    print(f"\n✅ Quarantined {success}/{len(legacy_runs)} runs")
    print("Legacy runs are now in runs_quarantine/ with reason='legacy_no_provenance'")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
