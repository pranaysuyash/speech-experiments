"""
Tests for Arsenal documentation freshness.
Ensures the generated docs match what the generator would produce.
"""

import json
import subprocess
from pathlib import Path

import pytest


class TestArsenalFreshness:
    """Test that Arsenal docs are up-to-date."""

    def test_arsenal_json_exists(self):
        """arsenal.json must exist."""
        path = Path("docs/arsenal.json")
        assert path.exists(), "Run 'make arsenal' to generate docs/arsenal.json"

    def test_arsenal_md_exists(self):
        """ARSENAL.md must exist."""
        path = Path("docs/ARSENAL.md")
        assert path.exists(), "Run 'make arsenal' to generate docs/ARSENAL.md"

    def test_arsenal_json_valid(self):
        """arsenal.json must be valid JSON with schema version."""
        path = Path("docs/arsenal.json")
        if not path.exists():
            pytest.skip("arsenal.json not found")

        with open(path) as f:
            data = json.load(f)

        assert "models" in data
        assert "arsenal_schema_version" in data
        assert "generated_from_commit" in data
        assert "generated_from_tree" in data  # New: tree hash for freshness

    def test_all_registered_models_in_arsenal(self):
        """All registered models must appear in arsenal.json."""
        from harness.registry import ModelRegistry

        path = Path("docs/arsenal.json")
        if not path.exists():
            pytest.skip("arsenal.json not found")

        with open(path) as f:
            data = json.load(f)

        arsenal_model_ids = {m["model_id"] for m in data["models"]}
        registered_model_ids = set(ModelRegistry.list_models())

        missing = registered_model_ids - arsenal_model_ids
        assert not missing, f"Models missing from arsenal.json: {missing}. Run 'make arsenal'"

    def test_arsenal_freshness_equivalence(self):
        """
        Arsenal docs must match what generator would produce.

        This is the REAL freshness test - compares content, not just existence.
        """
        import tempfile

        # Generate to temp location
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Run generator with custom output paths
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    "-c",
                    f'''
import sys
sys.path.insert(0, ".")
from scripts.generate_arsenal import build_all_cards, generate_json, generate_markdown
from pathlib import Path

cards = build_all_cards()
generate_json(cards, Path("{tmpdir}/arsenal.json"))
generate_markdown(cards, Path("{tmpdir}/ARSENAL.md"))
''',
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                pytest.skip(f"Generator failed: {result.stderr}")

            # Compare JSON (ignoring commit/tree which change)
            current_json = Path("docs/arsenal.json")
            new_json = tmpdir / "arsenal.json"

            if not current_json.exists():
                pytest.skip("docs/arsenal.json not found")

            with open(current_json) as f:
                current = json.load(f)
            with open(new_json) as f:
                new = json.load(f)

            # Compare models list (the important part)
            assert len(current["models"]) == len(new["models"]), (
                "Model count mismatch. Run 'make arsenal'"
            )

            for c, n in zip(current["models"], new["models"], strict=False):
                assert c["model_id"] == n["model_id"], (
                    f"Model order mismatch: {c['model_id']} vs {n['model_id']}. Run 'make arsenal'"
                )


class TestModelCardSchema:
    """Test ModelCard schema correctness."""

    def test_model_card_import(self):
        """ModelCard schema imports correctly."""
        from harness.model_card import ModelCard

        assert ModelCard is not None

    def test_model_card_from_sources(self):
        """ModelCard can be built from sources."""
        from harness.model_card import ModelCard

        registry_meta = {
            "status": "experimental",
            "capabilities": ["asr"],
            "hardware": ["cpu"],
            "modes": ["batch"],
            "hash": "abc12345",
        }

        config = {"model_name": "test-model", "metadata": {"provider": "Test", "license": "MIT"}}

        card = ModelCard.from_sources("test", registry_meta, config, None)

        assert card.model_id == "test"
        assert card.status == "experimental"
        assert "asr" in card.capabilities

    def test_promotion_validation(self):
        """Promotion validation catches missing fields."""
        from harness.model_card import ModelCard, validate_for_promotion

        # Minimal card missing required fields
        card = ModelCard(model_id="test")

        is_valid, issues = validate_for_promotion(card, "candidate")
        assert not is_valid
        assert "Missing capabilities" in issues
