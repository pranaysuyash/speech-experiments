from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.registry import ModelRegistry


def test_hf_sprint_models_are_registered():
    cfg_path = ROOT / "config" / "hf_sprint_2026q1.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    sprint_models: set[str] = set()
    for agent in cfg.get("agents", {}).values():
        sprint_models.update(agent.get("include_models", []))

    registered = set(ModelRegistry.list_models())
    missing = sorted(sprint_models - registered)
    assert missing == [], f"Unregistered sprint models: {missing}"
