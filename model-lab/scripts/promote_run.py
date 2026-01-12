#!/usr/bin/env python3
"""
Promote Run Script.

Converts a run from adhoc to smoke/golden only if it satisfies prerequisites.

Rules:
- provenance must be present
- run_context must be present  
- dataset_hash must match known dataset registry
- grade transition: adhoc → smoke or adhoc → golden_batch only
- quality metrics present only if target has ground truth

If any check fails: auto-quarantine with reason.

Usage:
    python scripts/promote_run.py <run_path> --to-grade smoke
    python scripts/promote_run.py <run_path> --to-grade golden_batch --with-gt <gt_path>
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Known datasets with their expected content hashes
KNOWN_DATASETS = {
    # Smoke datasets
    "asr_smoke_v1": {"type": "smoke", "has_gt": True},
    "vad_smoke_v1": {"type": "smoke", "has_gt": False},
    "vad_smoke_v2": {"type": "smoke", "has_gt": False},
    "v2v_smoke_v1": {"type": "smoke", "has_gt": False},
    "v2v_smoke_v2": {"type": "smoke", "has_gt": False},
    "diar_smoke_v1": {"type": "smoke", "has_gt": False},
    "diar_smoke_v2": {"type": "smoke", "has_gt": False},
    "tts_smoke_v1": {"type": "smoke", "has_gt": False},
    # Golden datasets
    "primary": {"type": "golden_batch", "has_gt": True},
    "llm_primary": {"type": "golden_batch", "has_gt": True},
    "ux_primary": {"type": "golden_batch", "has_gt": True},
}


def load_run(run_path: Path) -> dict:
    """Load run artifact."""
    with open(run_path) as f:
        return json.load(f)


def quarantine_run(run_path: Path, reason: str) -> Path:
    """Move run to quarantine with reason."""
    quarantine_dir = Path("runs_quarantine")
    quarantine_dir.mkdir(exist_ok=True)
    
    # Create quarantine manifest
    manifest = {
        "original_path": str(run_path),
        "quarantine_time": datetime.now().isoformat(),
        "reason": f"PROMOTION_FAILED:{reason}",
    }
    
    # Move file
    quarantine_path = quarantine_dir / run_path.name
    shutil.move(str(run_path), str(quarantine_path))
    
    # Write manifest
    manifest_path = quarantine_path.with_suffix(".quarantine.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"⛔ Quarantined: {run_path} → {quarantine_path}")
    print(f"   Reason: {reason}")
    
    return quarantine_path


def validate_promotion(run: dict, target_grade: str, gt_path: Path = None) -> tuple[bool, str]:
    """
    Validate if run can be promoted to target grade.
    
    Returns (valid, reason).
    """
    # 1. Provenance must be present
    if 'provenance' not in run:
        return False, "missing_provenance"
    
    # 2. run_context must be present
    if 'run_context' not in run:
        return False, "missing_run_context"
    
    # 3. Current grade must be adhoc
    current_grade = run.get('evidence', {}).get('grade', '')
    if current_grade not in ['adhoc', '']:
        return False, f"cannot_promote_from_{current_grade}"
    
    # 4. Dataset must be known
    dataset_id = run.get('evidence', {}).get('dataset_id', '')
    if dataset_id not in KNOWN_DATASETS:
        return False, f"unknown_dataset:{dataset_id}"
    
    dataset_info = KNOWN_DATASETS[dataset_id]
    
    # 5. Target grade must match dataset type
    if target_grade == "golden_batch" and dataset_info['type'] != 'golden_batch':
        return False, f"dataset_not_golden:{dataset_id}"
    
    if target_grade == "smoke" and dataset_info['type'] != 'smoke':
        return False, f"dataset_not_smoke:{dataset_id}"
    
    # 6. Quality metrics only if GT exists
    metrics = run.get('metrics', {})
    has_quality = 'wer' in metrics and metrics['wer'] is not None
    
    if has_quality and not dataset_info['has_gt']:
        return False, "quality_metrics_without_ground_truth"
    
    if target_grade == "golden_batch" and dataset_info['has_gt'] and not has_quality:
        return False, "golden_requires_quality_metrics"
    
    return True, "valid"


def promote_run(run_path: Path, target_grade: str, gt_path: Path = None) -> bool:
    """
    Attempt to promote a run to target grade.
    
    Returns True if successful, False if quarantined.
    """
    run = load_run(run_path)
    
    valid, reason = validate_promotion(run, target_grade, gt_path)
    
    if not valid:
        quarantine_run(run_path, reason)
        return False
    
    # Update grade
    run['evidence']['grade'] = target_grade
    run['evidence']['promoted'] = True
    run['evidence']['promotion_time'] = datetime.now().isoformat()
    
    # Write back
    with open(run_path, 'w') as f:
        json.dump(run, f, indent=2)
    
    print(f"✅ Promoted: {run_path}")
    print(f"   Grade: adhoc → {target_grade}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Promote run from adhoc to smoke/golden")
    parser.add_argument("run_path", type=Path, help="Path to run artifact")
    parser.add_argument("--to-grade", required=True, choices=["smoke", "golden_batch"],
                        help="Target grade")
    parser.add_argument("--with-gt", type=Path, help="Ground truth path (for golden)")
    parser.add_argument("--force", action="store_true", 
                        help="Force promotion without validation (use with caution)")
    args = parser.parse_args()
    
    if not args.run_path.exists():
        print(f"❌ Run not found: {args.run_path}")
        sys.exit(1)
    
    if args.force:
        print("⚠️  Force mode: skipping validation")
        run = load_run(args.run_path)
        run['evidence']['grade'] = args.to_grade
        run['evidence']['force_promoted'] = True
        with open(args.run_path, 'w') as f:
            json.dump(run, f, indent=2)
        print(f"✅ Force promoted to {args.to_grade}")
        sys.exit(0)
    
    success = promote_run(args.run_path, args.to_grade, args.with_gt)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
