"""
Bench runner tests (LCS-B1).

Tests for benchmark runner functions.
"""

import pytest
from unittest.mock import MagicMock, patch


from bench.runner import (
    get_run_id,
    create_result_schema,
    format_result_table,
)


class TestGetRunId:
    """Test run_id generation."""
    
    def test_run_id_format(self):
        """Run ID should be date_time_uuid format."""
        rid = get_run_id()
        assert isinstance(rid, str)
        assert "_" in rid
        # Should have format YYYYMMDD_HHMMSS_hex
        parts = rid.split("_")
        assert len(parts) == 3


class TestCreateResultSchema:
    """Test result schema creation."""
    
    def test_creates_valid_schema(self):
        """Should create valid result schema."""
        result = create_result_schema(
            model_id="test_model",
            surface="asr",
            input_info={"path": "test.wav", "duration_s": 5.0, "sr": 16000},
            metrics={"wer": 0.1, "cer": 0.05},
            timing={"wall_s": 2.5, "rtf": 0.5},
            env={"device": "cpu", "runtime": "pytorch"},
        )
        
        # Required fields
        assert "run_id" in result
        assert "timestamp" in result
        assert result["model_id"] == "test_model"
        assert result["surface"] == "asr"
        assert result["input"]["path"] == "test.wav"
        assert result["metrics"]["wer"] == 0.1
        assert result["timing"]["rtf"] == 0.5
        assert result["env"]["device"] == "cpu"


class TestFormatResultTable:
    """Test table formatting."""
    
    def test_empty_results(self):
        """Empty list returns message."""
        result = format_result_table([])
        assert "No results" in result
    
    def test_formats_results(self):
        """Formats multiple results."""
        results = [
            {
                "model_id": "model_a",
                "surface": "asr",
                "metrics": {"wer": 0.1, "cer": 0.05, "rtf": 0.5},
                "timing": {"rtf": 0.5},
            },
            {
                "model_id": "model_b",
                "surface": "asr_stream",
                "metrics": {"wer": 0.15, "cer": 0.08, "rtf": 0.3, "first_token_latency_ms": 45.2},
                "timing": {"rtf": 0.3},
            },
        ]
        
        table = format_result_table(results)
        
        assert "model_a" in table
        assert "model_b" in table
        assert "asr" in table
        assert "asr_stream" in table
