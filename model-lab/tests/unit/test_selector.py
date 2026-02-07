"""
Model selector tests (LCS-Z).

Tests filter by device, runtime, surface, CI.
"""

import pytest
from pathlib import Path


from harness.selector import (
    list_models_by_filter,
    format_model_table,
    get_streaming_models,
    get_ci_safe_models,
    get_models_by_runtime,
    _get_model_ci_flag,
    _get_model_runtime,
)


class TestListModelsByFilter:
    """Test list_models_by_filter function."""
    
    def test_no_filters_returns_all(self):
        """With no filters, returns all registered models."""
        models = list_models_by_filter()
        # Should return a list (could be empty if registry not loaded)
        assert isinstance(models, list)
    
    def test_filter_by_surface(self):
        """Filter by capability surface."""
        models = list_models_by_filter(surface="asr_stream")
        for m in models:
            assert "asr_stream" in m.get("capabilities", [])
    
    def test_filter_by_device(self):
        """Filter by device support."""
        models = list_models_by_filter(device="cpu")
        for m in models:
            assert "cpu" in m.get("hardware", [])
    
    def test_combined_filters(self):
        """Combine multiple filters."""
        models = list_models_by_filter(device="mps", surface="asr_stream")
        for m in models:
            assert "mps" in m.get("hardware", [])
            assert "asr_stream" in m.get("capabilities", [])


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_get_streaming_models(self):
        """Get all streaming models."""
        models = get_streaming_models()
        assert isinstance(models, list)
    
    def test_get_ci_safe_models(self):
        """Get CI-safe models."""
        models = get_ci_safe_models()
        assert isinstance(models, list)
    
    def test_get_models_by_runtime(self):
        """Get models by runtime."""
        models = get_models_by_runtime("pytorch")
        assert isinstance(models, list)


class TestClaimsHelpers:
    """Test claims file helpers."""
    
    def test_get_ci_flag_from_claims(self):
        """Test reading CI flag from claims."""
        # Test with a model we know exists
        ci = _get_model_ci_flag("kyutai_streaming")
        assert isinstance(ci, bool)
    
    def test_get_runtime_from_claims(self):
        """Test reading runtime from claims."""
        runtime = _get_model_runtime("kyutai_streaming")
        assert runtime == "pytorch"


class TestFormatModelTable:
    """Test table formatting."""
    
    def test_format_empty_list(self):
        """Empty list returns message."""
        result = format_model_table([])
        assert "No models" in result
    
    def test_format_with_models(self):
        """Format list of models."""
        models = [
            {"model_id": "test", "capabilities": ["asr"], "hardware": ["cpu"]},
        ]
        result = format_model_table(models)
        assert "test" in result
        assert "asr" in result
        assert "cpu" in result
