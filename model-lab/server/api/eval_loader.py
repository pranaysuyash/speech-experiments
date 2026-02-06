"""
Eval loading helper with canonical location preference.

Canonical: run_dir/eval.json
Fallback: run_dir/bundle/eval.json (legacy)
"""

import json
from pathlib import Path
from typing import Any


def load_eval(run_root: Path) -> dict[str, Any] | None:
    """
    Load eval.json from run directory.

    Prefers run_root/eval.json (canonical).
    Falls back to run_root/bundle/eval.json (legacy).

    Returns None if not found or malformed.
    """
    # Try canonical location first
    for eval_path in [run_root / "eval.json", run_root / "bundle" / "eval.json"]:
        if not eval_path.exists():
            continue
        try:
            return json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception:
            # Malformed JSON, try next location
            continue

    return None
