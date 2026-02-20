#!/usr/bin/env python3
"""
Prefetch Hugging Face Hub assets for the sprint.

This is mainly about de-risking the March 1 deadline:
- verify gated access works (early)
- warm HF cache for repeatable runs

Defaults to a "small" strategy (cards/config/tokenizers) to avoid accidentally
pulling multi-GB weights.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.env import load_dotenv_if_present


HF_URL_RE = re.compile(r"^https?://huggingface\\.co/(?P<repo>[^/]+/[^/?#]+)")


@dataclass(frozen=True)
class RepoRef:
    model_id: str
    repo_id: str


def _infer_repo_id(model_id: str) -> str | None:
    cfg_path = PROJECT_ROOT / "models" / model_id / "config.yaml"
    if not cfg_path.exists():
        return None

    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None

    # Common schemas in this repo:
    # - {"model_name": "org/name"} (sometimes)
    # - {"metadata": {"model_name": "org/name", "repo_url": "https://huggingface.co/org/name"}}
    # - {"source": {"repo": "org/name", "hub": "https://huggingface.co/org/name"}}
    for key in ("model_name",):
        v = cfg.get(key)
        if isinstance(v, str) and "/" in v and " " not in v:
            return v

    meta = cfg.get("metadata") or {}
    if isinstance(meta, dict):
        v = meta.get("model_name")
        if isinstance(v, str) and "/" in v and " " not in v:
            return v
        repo_url = meta.get("repo_url")
        if isinstance(repo_url, str):
            m = HF_URL_RE.match(repo_url.strip())
            if m:
                return m.group("repo")

    src = cfg.get("source") or {}
    if isinstance(src, dict):
        v = src.get("repo")
        if isinstance(v, str) and "/" in v and " " not in v:
            return v
        hub = src.get("hub")
        if isinstance(hub, str):
            m = HF_URL_RE.match(hub.strip())
            if m:
                return m.group("repo")

    return None


def _load_model_ids_from_config(config_path: Path) -> list[str]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    agents = cfg.get("agents") or {}
    model_ids: list[str] = []
    for agent_cfg in agents.values():
        mids = (agent_cfg or {}).get("include_models") or []
        for mid in mids:
            model_ids.append(str(mid))
    return sorted(set(model_ids))


def _download_small(repo_id: str, *, token: str | None, cache_dir: Path | None) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    downloaded: list[str] = []
    attempted = [
        "README.md",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ]

    for filename in attempted:
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=token,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
            downloaded.append(path)
        except Exception:
            continue

    return {"attempted": attempted, "downloaded": downloaded}


def main() -> int:
    load_dotenv_if_present()

    parser = argparse.ArgumentParser(description="Prefetch HF Hub assets for sprint models")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/hf_sprint_2026q1.yaml"),
        help="Sprint config YAML",
    )
    parser.add_argument(
        "--strategy",
        choices=["small", "card"],
        default="small",
        help="Download strategy (default: small)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".huggingface"),
        help="Cache dir to use (default: .huggingface)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/hf_sprint_2026q1/prefetch/prefetch.json"),
        help="Output JSON report path",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    model_ids = _load_model_ids_from_config(args.config)
    repos: list[RepoRef] = []
    for model_id in model_ids:
        repo_id = _infer_repo_id(model_id)
        if repo_id:
            repos.append(RepoRef(model_id=model_id, repo_id=repo_id))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "config": str(args.config),
        "strategy": args.strategy,
        "cache_dir": str(args.cache_dir),
        "models_total": len(model_ids),
        "repos_inferred": len(repos),
        "results": [],
    }

    for ref in repos:
        row: dict[str, Any] = {"model_id": ref.model_id, "repo_id": ref.repo_id}
        try:
            if args.strategy == "card":
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(
                    repo_id=ref.repo_id,
                    filename="README.md",
                    token=token,
                    cache_dir=str(args.cache_dir),
                )
                row["status"] = "ok"
                row["downloaded"] = [path]
            else:
                row["status"] = "ok"
                row["downloads"] = _download_small(
                    ref.repo_id, token=token, cache_dir=args.cache_dir
                )
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        report["results"].append(row)

    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    ok = sum(1 for r in report["results"] if r.get("status") == "ok")
    print(f"✓ Prefetch complete: ok={ok} total={len(report['results'])}")
    print(f"Report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

