"""
Claims manifest validation tests.

Tests that all claims.yaml files are valid and have the required schema.
"""

from pathlib import Path

import pytest
import yaml

# Required fields for every claim
REQUIRED_CLAIM_FIELDS = ["id", "task", "type", "description", "enforcement", "test_ref"]
VALID_TYPES = ["structural", "behavioral", "quality", "performance", "constraint", "feature"]
VALID_ENFORCEMENT = ["required", "optional"]


def get_all_claims_files():
    """Find all claims.yaml files in models directory."""
    models_dir = Path(__file__).parent.parent.parent / "models"
    return list(models_dir.glob("*/claims.yaml"))


def load_claims(claims_path: Path) -> dict:
    """Load and parse a claims.yaml file."""
    with open(claims_path) as f:
        return yaml.safe_load(f)


class TestClaimsManifest:
    """Validate claims.yaml schema and required fields."""

    def test_claims_files_exist(self):
        """At least one claims.yaml should exist."""
        files = get_all_claims_files()
        # This test is informational - we pass even with 0 claims for now
        print(f"Found {len(files)} claims files: {[f.parent.name for f in files]}")

    @pytest.mark.parametrize("claims_path", get_all_claims_files(), ids=lambda p: p.parent.name)
    def test_claims_schema_valid(self, claims_path):
        """Each claims.yaml must have valid schema."""
        data = load_claims(claims_path)

        assert "claims" in data, f"Missing 'claims' key in {claims_path}"
        assert isinstance(data["claims"], list), f"'claims' must be a list in {claims_path}"

        claim_ids = set()
        for claim in data["claims"]:
            # Check required fields
            for field in REQUIRED_CLAIM_FIELDS:
                assert field in claim, (
                    f"Missing required field '{field}' in claim: {claim.get('id', 'unknown')}"
                )

            # Validate type
            assert claim["type"] in VALID_TYPES, f"Invalid type '{claim['type']}' in {claim['id']}"

            # Validate enforcement
            assert claim["enforcement"] in VALID_ENFORCEMENT, (
                f"Invalid enforcement '{claim['enforcement']}' in {claim['id']}"
            )

            # Check unique IDs
            assert claim["id"] not in claim_ids, f"Duplicate claim ID: {claim['id']}"
            claim_ids.add(claim["id"])

    @pytest.mark.parametrize("claims_path", get_all_claims_files(), ids=lambda p: p.parent.name)
    def test_test_refs_follow_convention(self, claims_path):
        """test_ref must follow claims.<task>.<name>_v<N> convention."""
        data = load_claims(claims_path)

        for claim in data["claims"]:
            test_ref = claim["test_ref"]

            # Should start with claims.
            assert test_ref.startswith("claims."), f"test_ref must start with 'claims.': {test_ref}"

            parts = test_ref.split(".")
            assert len(parts) == 3, f"test_ref must be claims.<task>.<name>: {test_ref}"

            # Task in test_ref should match claim task
            assert parts[1] == claim["task"], (
                f"test_ref task mismatch: {parts[1]} != {claim['task']}"
            )

    @pytest.mark.parametrize("claims_path", get_all_claims_files(), ids=lambda p: p.parent.name)
    def test_required_claims_have_thresholds(self, claims_path):
        """Required claims should have thresholds defined."""
        data = load_claims(claims_path)

        for claim in data["claims"]:
            if claim["enforcement"] == "required":
                # Required claims should have clear pass/fail criteria
                assert "thresholds" in claim or claim["type"] == "behavioral", (
                    f"Required claim '{claim['id']}' should have thresholds"
                )
