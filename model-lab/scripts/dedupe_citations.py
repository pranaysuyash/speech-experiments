#!/usr/bin/env python3
"""
Dedupe a raw citations markdown/text file into a CSV with URL counts.

Purpose:
- The chat-provided citations lists often contain many duplicates.
- This script produces a stable, machine-readable bibliography to attach to research notes.

Heuristics:
- A "citation" is detected via http(s):// URLs.
- Title is the nearest non-empty non-URL line immediately above a URL (best effort).
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


URL_RE = re.compile(r"(https?://\S+)")


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _extract_pairs(lines: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    last_title = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        m = URL_RE.search(line)
        if m:
            url = m.group(1).rstrip(").,]")
            title = last_title.strip()
            pairs.append((url, title))
            continue

        # Treat non-URL lines as potential titles.
        if not line.startswith("#") and "http" not in line:
            last_title = line
    return pairs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input markdown/text with URLs")
    p.add_argument("--output", required=True, help="Output CSV path")
    args = p.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    lines = _read_lines(in_path)
    pairs = _extract_pairs(lines)
    url_counts = Counter([u for u, _ in pairs])
    url_titles: dict[str, list[str]] = defaultdict(list)
    for url, title in pairs:
        if title:
            url_titles[url].append(title)

    rows = []
    for url, count in url_counts.most_common():
        titles = url_titles.get(url, [])
        # Keep the first title we saw; retain alternates as a semicolon string.
        primary = titles[0] if titles else ""
        alternates = "; ".join(t for t in titles[1:] if t and t != primary)
        rows.append(
            {
                "url": url,
                "count": str(count),
                "title": primary,
                "alternate_titles": alternates,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["url", "count", "title", "alternate_titles"], extrasaction="ignore"
        )
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} deduped URLs to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
