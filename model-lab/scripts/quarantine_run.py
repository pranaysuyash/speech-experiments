#!/usr/bin/env python3
"""
Quarantine helper - safely move runs to quarantine with audit trail.

Usage:
    python scripts/quarantine_run.py runs/model/task/file.json --reason "description"
    python scripts/quarantine_run.py --list  # Show manifest
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


QUARANTINE_DIR = Path("runs_quarantine")
MANIFEST_FILE = QUARANTINE_DIR / "manifest.json"


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of file contents."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def load_manifest() -> list:
    """Load quarantine manifest."""
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE) as f:
            return json.load(f)
    return []


def save_manifest(manifest: list):
    """Save quarantine manifest."""
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def quarantine_run(run_path: Path, reason: str) -> bool:
    """
    Move a run to quarantine with audit entry.
    
    Returns True if successful, False if already quarantined.
    """
    if not run_path.exists():
        print(f"❌ File not found: {run_path}")
        return False
    
    # Compute hash before moving
    file_hash = compute_file_hash(run_path)
    
    # Determine destination
    # Preserve relative path structure under quarantine
    try:
        rel_path = run_path.relative_to("runs")
    except ValueError:
        rel_path = run_path.name
    
    dest_path = QUARANTINE_DIR / rel_path
    
    # Check if already quarantined
    manifest = load_manifest()
    for entry in manifest:
        if entry.get("file_hash") == file_hash:
            print(f"⚠️  Already quarantined: {entry['original_path']}")
            return False
    
    # Move file
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_path), str(dest_path))
    
    # Add manifest entry
    entry = {
        "original_path": str(run_path),
        "quarantine_path": str(dest_path),
        "reason": reason,
        "file_hash": file_hash,
        "quarantined_at": datetime.now().isoformat(),
        "quarantined_by": "agent"
    }
    manifest.append(entry)
    save_manifest(manifest)
    
    print(f"✅ Quarantined: {run_path}")
    print(f"   Reason: {reason}")
    print(f"   Hash: {file_hash}")
    print(f"   Dest: {dest_path}")
    
    return True


def list_quarantine():
    """List all quarantined runs."""
    manifest = load_manifest()
    
    if not manifest:
        print("No quarantined runs.")
        return
    
    print(f"\n{'='*70}")
    print(f"  Quarantined Runs ({len(manifest)})")
    print(f"{'='*70}\n")
    
    for entry in manifest:
        print(f"  📁 {entry['original_path']}")
        print(f"     Reason: {entry['reason']}")
        print(f"     Hash: {entry['file_hash']}")
        print(f"     When: {entry['quarantined_at']}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Quarantine runs with audit trail")
    parser.add_argument("run_path", type=str, nargs="?", help="Path to run file")
    parser.add_argument("--reason", type=str, default="unspecified", 
                       help="Reason for quarantine")
    parser.add_argument("--list", action="store_true", help="List quarantined runs")
    
    args = parser.parse_args()
    
    if args.list:
        list_quarantine()
        return 0
    
    if not args.run_path:
        parser.error("Either provide a run_path or use --list")
    
    run_path = Path(args.run_path)
    if quarantine_run(run_path, args.reason):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
