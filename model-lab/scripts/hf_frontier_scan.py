#!/usr/bin/env python3
"""
Scan Hugging Face Hub for frontier speech/audio models and runtime signals.

Focuses on:
- realtime/streaming speech models
- transformers-first model repos
- MLX-compatible repos
- GGUF / llama.cpp-friendly repos
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


DEFAULT_QUERIES = [
    "voxtral",
    "realtime speech",
    "streaming asr",
    "mlx whisper",
    "gguf whisper",
    "llama.cpp asr",
]


@dataclass
class Row:
    query: str
    model_id: str
    pipeline_tag: str | None
    last_modified: str | None
    downloads: int | None
    likes: int | None
    runtime_hints: list[str]
    tags: list[str]


def _runtime_hints(tags: list[str]) -> list[str]:
    normalized = {t.lower() for t in tags}
    hints: list[str] = []
    if "transformers" in normalized:
        hints.append("transformers")
    if "mlx" in normalized or any(t.startswith("mlx") for t in normalized):
        hints.append("mlx")
    if "gguf" in normalized:
        hints.append("gguf")
    if "llama.cpp" in normalized or "llamacpp" in normalized:
        hints.append("llama.cpp")
    if "realtime" in normalized or "streaming" in normalized:
        hints.append("realtime/streaming")
    return sorted(set(hints))


def run_scan(queries: list[str], *, limit: int) -> list[Row]:
    api = HfApi()
    rows: list[Row] = []

    for q in queries:
        results = api.list_models(search=q, sort="lastModified", direction=-1, limit=limit)
        for m in results:
            tags = list(m.tags or [])
            rows.append(
                Row(
                    query=q,
                    model_id=m.id,
                    pipeline_tag=getattr(m, "pipeline_tag", None),
                    last_modified=m.last_modified.isoformat()
                    if getattr(m, "last_modified", None)
                    else None,
                    downloads=getattr(m, "downloads", None),
                    likes=getattr(m, "likes", None),
                    runtime_hints=_runtime_hints(tags),
                    tags=tags[:20],
                )
            )
    return rows


def dedupe(rows: list[Row]) -> list[Row]:
    best: dict[str, Row] = {}
    for r in rows:
        cur = best.get(r.model_id)
        if cur is None:
            best[r.model_id] = r
            continue

        cur_score = (cur.downloads or 0, cur.likes or 0)
        new_score = (r.downloads or 0, r.likes or 0)
        if new_score > cur_score:
            best[r.model_id] = r
    return sorted(
        best.values(),
        key=lambda x: ((x.downloads or 0), (x.likes or 0), x.last_modified or ""),
        reverse=True,
    )


def write_markdown(path: Path, rows: list[Row]) -> None:
    lines: list[str] = []
    lines.append("# HF Frontier Scan")
    lines.append("")
    lines.append(f"Generated: {datetime.now(UTC).isoformat()}")
    lines.append("")
    lines.append("| model_id | runtime_hints | pipeline_tag | downloads | likes | last_modified |")
    lines.append("|---|---|---|---:|---:|---|")
    for r in rows:
        hints = ",".join(r.runtime_hints) if r.runtime_hints else "-"
        lines.append(
            f"| `{r.model_id}` | {hints} | {r.pipeline_tag or '-'} | "
            f"{r.downloads or 0} | {r.likes or 0} | {r.last_modified or '-'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan HF Hub for frontier speech models")
    parser.add_argument("--limit", type=int, default=50, help="Max models per query")
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Search query (repeatable). If omitted, built-in queries are used.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("runs/hf_sprint_2026q1/frontier/frontier_scan.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("runs/hf_sprint_2026q1/frontier/frontier_scan.md"),
        help="Output Markdown path",
    )
    args = parser.parse_args()

    queries = args.query if args.query else DEFAULT_QUERIES
    raw_rows = run_scan(queries, limit=args.limit)
    rows = dedupe(raw_rows)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "queries": queries,
        "count_raw": len(raw_rows),
        "count_deduped": len(rows),
        "rows": [asdict(r) for r in rows],
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(args.out_md, rows)

    print(f"✓ Frontier scan complete: raw={len(raw_rows)} deduped={len(rows)}")
    print(f"JSON: {args.out_json}")
    print(f"MD: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
