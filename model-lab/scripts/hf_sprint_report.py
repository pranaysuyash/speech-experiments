#!/usr/bin/env python3
"""
Aggregate HF sprint execution ledgers into machine-readable and human-readable reports.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _extract_metrics(artifact: dict[str, Any]) -> dict[str, Any]:
    metrics = artifact.get("metrics", {}) if isinstance(artifact, dict) else {}
    quality = artifact.get("metrics_quality", {}) if isinstance(artifact, dict) else {}
    structural = artifact.get("metrics_structural", {}) if isinstance(artifact, dict) else {}
    timing = artifact.get("timing", {}) if isinstance(artifact, dict) else {}

    def first_non_null(*vals: Any) -> Any:
        for v in vals:
            if v is not None:
                return v
        return None

    return {
        "wer": first_non_null(quality.get("wer"), metrics.get("wer")),
        "cer": first_non_null(quality.get("cer"), metrics.get("cer")),
        "rtf": first_non_null(structural.get("rtf"), metrics.get("rtf"), timing.get("rtf")),
        "latency_ms": first_non_null(
            structural.get("latency_ms"),
            metrics.get("latency_ms"),
            metrics.get("wall_ms"),
            timing.get("wall_ms"),
        ),
        "first_token_latency_ms": first_non_null(
            metrics.get("first_token_latency_ms"),
            structural.get("first_token_latency_ms"),
        ),
    }


def _extract_model_id(entry: dict[str, Any], artifact: dict[str, Any] | None) -> str:
    task_model = entry.get("task", {}).get("model_id")
    if task_model:
        return str(task_model)
    if not artifact:
        return "unknown"
    return str(
        artifact.get("model_id")
        or artifact.get("provider_id")
        or artifact.get("meta", {}).get("model_id")
        or "unknown"
    )


def load_execution_rows(execution_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for ledger_path in sorted(execution_root.glob("*/ledger.jsonl")):
        agent_id = ledger_path.parent.name
        for entry in _read_jsonl(ledger_path):
            artifact_path = entry.get("artifact_path")
            artifact = None
            if isinstance(artifact_path, str) and artifact_path:
                path = Path(artifact_path)
                if path.exists():
                    try:
                        artifact = json.loads(path.read_text(encoding="utf-8"))
                    except Exception:
                        artifact = None

            metrics = _extract_metrics(artifact or {})
            model_id = _extract_model_id(entry, artifact)

            task = entry.get("task", {})
            rows.append(
                {
                    "agent_id": agent_id,
                    "timestamp": entry.get("timestamp"),
                    "status": entry.get("status"),
                    "exit_code": entry.get("exit_code"),
                    "duration_s": entry.get("duration_s"),
                    "task_id": task.get("task_id"),
                    "model_id": model_id,
                    "capability": task.get("capability"),
                    "dataset": task.get("dataset"),
                    "artifact_path": artifact_path,
                    "wer": metrics["wer"],
                    "cer": metrics["cer"],
                    "rtf": metrics["rtf"],
                    "latency_ms": metrics["latency_ms"],
                    "first_token_latency_ms": metrics["first_token_latency_ms"],
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return

    cols = [
        "agent_id",
        "timestamp",
        "status",
        "exit_code",
        "duration_s",
        "task_id",
        "model_id",
        "capability",
        "dataset",
        "artifact_path",
        "wer",
        "cer",
        "rtf",
        "latency_ms",
        "first_token_latency_ms",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(rows: list[dict[str, Any]], out_path: Path, sprint_id: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat()

    by_agent = defaultdict(list)
    for row in rows:
        by_agent[row["agent_id"]].append(row)

    model_wer: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["wer"] is None:
            continue
        try:
            model_wer[row["model_id"]].append(float(row["wer"]))
        except Exception:
            continue

    lines: list[str] = []
    lines.append(f"# HF Sprint Report: {sprint_id}")
    lines.append("")
    lines.append(f"Generated at: {now}")
    lines.append("")
    lines.append(f"Total ledger rows: {len(rows)}")
    lines.append("")
    lines.append("## Agent Summary")
    lines.append("")
    lines.append("| agent_id | total | ok | failed | manual_skips |")
    lines.append("|---|---:|---:|---:|---:|")

    for agent_id in sorted(by_agent):
        items = by_agent[agent_id]
        ok = sum(1 for x in items if x["status"] == "ok")
        failed = sum(1 for x in items if x["status"] == "failed")
        manual = sum(1 for x in items if x["status"] == "skipped_manual")
        lines.append(f"| {agent_id} | {len(items)} | {ok} | {failed} | {manual} |")

    lines.append("")
    lines.append("## Model WER (where available)")
    lines.append("")
    lines.append("| model_id | runs_with_wer | avg_wer |")
    lines.append("|---|---:|---:|")

    for model_id in sorted(model_wer):
        vals = model_wer[model_id]
        avg = sum(vals) / len(vals)
        lines.append(f"| {model_id} | {len(vals)} | {avg:.4f} |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate HF sprint report")
    parser.add_argument(
        "--execution-root",
        type=Path,
        default=Path("runs/hf_sprint_2026q1/execution"),
        help="Directory containing per-agent ledger.jsonl files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/hf_sprint_2026q1/reports"),
        help="Output directory for report files",
    )
    parser.add_argument(
        "--sprint-id",
        default="hf_pro_2026q1",
        help="Sprint identifier for report titles",
    )
    args = parser.parse_args()

    rows = load_execution_rows(args.execution_root)
    csv_path = args.out_dir / "task_results.csv"
    md_path = args.out_dir / "summary.md"

    write_csv(rows, csv_path)
    write_markdown_summary(rows, md_path, args.sprint_id)

    print(f"Report rows: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
