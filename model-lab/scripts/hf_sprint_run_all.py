#!/usr/bin/env python3
"""
Run the HF sprint end-to-end from one command:
1) Generate plan artifacts
2) (Optional) run preflight checks
3) Execute all agent queues in parallel
4) Generate aggregate report

This is designed for the "HF Pro window" where time-to-evidence matters.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> int:
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HF sprint (plan -> workers -> report)")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/hf_sprint_2026q1.yaml"),
        help="Sprint config YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/hf_sprint_2026q1"),
        help="Output directory for plan/execution/report",
    )
    parser.add_argument("--preflight", action="store_true", help="Run preflight checks first")
    parser.add_argument("--dry-run", action="store_true", help="Dry-run workers (no execution)")
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Execute at most N tasks per queue",
    )
    args = parser.parse_args()

    # 1) Plan
    rc = _run(
        [
            "uv",
            "run",
            "python",
            "scripts/hf_sprint_plan.py",
            "--config",
            str(args.config),
            "--output-dir",
            str(args.output_dir),
        ]
    )
    if rc != 0:
        return rc

    # 2) Preflight (optional)
    if args.preflight:
        rc = _run(["uv", "run", "python", "scripts/hf_sprint_preflight.py", "--config", str(args.config)])
        if rc != 0:
            return rc

    # 3) Run workers in parallel (one process per queue)
    queue_dir = args.output_dir / "agent_queues"
    queues = sorted(queue_dir.glob("*.json"))
    if not queues:
        print(f"❌ No queues found under: {queue_dir}")
        return 2

    procs: list[subprocess.Popen] = []
    for q in queues:
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/hf_sprint_worker.py",
            "--queue",
            str(q),
            "--execution-root",
            str(args.output_dir / "execution"),
            "--continue-on-error",
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.max_tasks is not None:
            cmd += ["--max-tasks", str(args.max_tasks)]
        procs.append(subprocess.Popen(cmd, cwd=PROJECT_ROOT))

    exit_codes = [p.wait() for p in procs]
    if any(code != 0 for code in exit_codes):
        print(f"⚠️  Some queues had failures: {exit_codes}")

    # 4) Report
    rc = _run(
        [
            "uv",
            "run",
            "python",
            "scripts/hf_sprint_report.py",
            "--execution-root",
            str(args.output_dir / "execution"),
            "--out-dir",
            str(args.output_dir / "reports"),
            "--sprint-id",
            "hf_pro_2026q1",
        ]
    )
    if rc != 0:
        return rc

    print(f"✓ Report: {args.output_dir / 'reports' / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

