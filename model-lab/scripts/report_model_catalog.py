#!/usr/bin/env python3
"""
Model catalog CSV reporter.

Takes the CSV schema from docs/research and makes it usable:
- Validates required columns.
- Prints a compact Markdown summary table.

This is intentionally lightweight; it does not mutate harness registries.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REQUIRED_COLS = [
    "model_id",
    "category",
    "tasks_supported",
    "local_or_api",
    "streaming_support",
    "license",
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")
        missing = [c for c in REQUIRED_COLS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        rows = []
        for r in reader:
            rows.append({k: (v or "").strip() for k, v in r.items()})
        return rows


def _md_escape(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ").strip()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        default=str(Path(__file__).resolve().parents[1] / "data" / "model_catalog.csv"),
        help="Path to model catalog CSV (default: model-lab/data/model_catalog.csv)",
    )
    p.add_argument("--limit", type=int, default=50)
    args = p.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(
            f"CSV not found: {csv_path}\n"
            f"Start from: model-lab/data/templates/model_catalog_schema.csv"
        )

    rows = _read_rows(csv_path)
    rows = [r for r in rows if r.get("model_id")]

    print(f"# Model Catalog Summary\n\nSource: `{csv_path}`\n\nTotal rows: {len(rows)}\n")
    print("| model_id | category | tasks | local/api | streaming | license |")
    print("|---|---|---|---|---|---|")
    for r in rows[: args.limit]:
        print(
            "| "
            + " | ".join(
                [
                    _md_escape(r.get("model_id", "")),
                    _md_escape(r.get("category", "")),
                    _md_escape(r.get("tasks_supported", "")),
                    _md_escape(r.get("local_or_api", "")),
                    _md_escape(r.get("streaming_support", "")),
                    _md_escape(r.get("license", "")),
                ]
            )
            + " |"
        )

    if len(rows) > args.limit:
        print(f"\n(Showing first {args.limit} rows.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
