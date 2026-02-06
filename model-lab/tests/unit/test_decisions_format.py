"""
Tests for DECISIONS.md output format integrity.

These tests prevent formatting regressions that would break:
- Machine parsing of decisions
- Auditability of metric values
"""

import re
from pathlib import Path

import pytest


class TestDecisionsFormat:
    """Tests that DECISIONS.md is machine-parseable."""

    def test_no_double_equals_in_metrics(self):
        """Metric cells must not have double-equals (e.g., latency=500=0.95)."""
        decisions_path = Path(__file__).parent.parent.parent / "docs" / "DECISIONS.md"
        if not decisions_path.exists():
            pytest.skip("DECISIONS.md not found")

        content = decisions_path.read_text()

        # Find all table cells with metrics (pattern: | metric |)
        metric_cells = re.findall(r"\| ([^|]+=[^|]+) \|", content)

        violations = []
        for cell in metric_cells:
            # Count equals signs - should be exactly 1 per metric
            equals_count = cell.count("=")
            if equals_count > 1:
                violations.append(f"Double-equals in metric: '{cell.strip()}'")

        if violations:
            pytest.fail("DECISIONS.md formatting errors:\n" + "\n".join(violations))

    def test_v2v_uses_rtf_like(self):
        """V2V metrics should use rtf_like, not raw latency_ms."""
        decisions_path = Path(__file__).parent.parent.parent / "docs" / "DECISIONS.md"
        if not decisions_path.exists():
            pytest.skip("DECISIONS.md not found")

        content = decisions_path.read_text()

        # Check V2V rows in pipeline tables
        v2v_rows = re.findall(r"\| v2v \| [^|]+ \| [^|]+ \| ([^|]+) \|", content)

        for metric in v2v_rows:
            metric = metric.strip()
            # Should be rtf_like=X.XX, not latency_ms=XXXX
            if "latency_ms" in metric and "rtf_like" not in metric:
                pytest.fail(f"V2V should use rtf_like, found: '{metric}'")

    def test_pipeline_tables_have_required_columns(self):
        """Pipeline tables must have Task, Model, Grade, Metric columns."""
        decisions_path = Path(__file__).parent.parent.parent / "docs" / "DECISIONS.md"
        if not decisions_path.exists():
            pytest.skip("DECISIONS.md not found")

        content = decisions_path.read_text()

        # Find table headers
        headers = re.findall(r"\| Task \| Model \| Grade \| Metric \|", content)

        # Should have at least one properly formatted table
        if not headers:
            # Check if there are any pipeline tables at all
            if "pipeline" in content.lower():
                pytest.fail("Pipeline tables missing required column format")
