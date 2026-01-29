"""Tests for candidate params merging.

These tests verify the deep merge logic without importing fastapi-dependent modules.
"""

import pytest
from typing import Any, Dict


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries (copy of implementation for testing).

    Override values take precedence. Nested dicts are merged recursively.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class TestDeepMerge:
    """Test deep merge utility."""

    def test_deep_merge_simple(self):
        """Test simple key merge."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)

        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge_nested(self):
        """Test nested dict merge."""
        base = {"asr": {"model_type": "whisper", "language": "en"}}
        override = {"asr": {"model_name": "base"}}
        result = _deep_merge(base, override)

        assert result == {"asr": {"model_type": "whisper", "language": "en", "model_name": "base"}}

    def test_deep_merge_override_nested(self):
        """Test that override takes precedence in nested dicts."""
        base = {"asr": {"model_type": "whisper", "model_name": "large-v3"}}
        override = {"asr": {"model_name": "base"}}
        result = _deep_merge(base, override)

        assert result == {"asr": {"model_type": "whisper", "model_name": "base"}}

    def test_deep_merge_empty_base(self):
        """Test merge with empty base."""
        result = _deep_merge({}, {"a": 1})
        assert result == {"a": 1}

    def test_deep_merge_empty_override(self):
        """Test merge with empty override."""
        result = _deep_merge({"a": 1}, {})
        assert result == {"a": 1}

    def test_deep_merge_preserves_base(self):
        """Test that original dicts are not mutated."""
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        result = _deep_merge(base, override)

        # Result has merged values
        assert result == {"a": {"b": 1, "c": 2}}
        # Original base unchanged
        assert base == {"a": {"b": 1}}

    def test_deep_merge_three_level_nesting(self):
        """Test merge with three levels of nesting."""
        base = {"level1": {"level2": {"level3": "base_value"}}}
        override = {"level1": {"level2": {"new_key": "override_value"}}}
        result = _deep_merge(base, override)

        assert result == {
            "level1": {
                "level2": {
                    "level3": "base_value",
                    "new_key": "override_value"
                }
            }
        }


class TestCandidateParamsMerging:
    """Test candidate params merge scenarios."""

    def test_config_merge_scenario(self):
        """Test realistic config merge scenario for ASR model comparison."""
        # Candidate defines model type and name
        candidate_params = {
            "asr": {"model_type": "whisper", "model_name": "base", "language": "en"}
        }

        # User overrides device preference from UI
        ui_config = {
            "device_preference": ["mps", "cpu"],
            "asr": {"language": "auto"}  # User wants auto-detect
        }

        result = _deep_merge(candidate_params, ui_config)

        # Candidate model settings preserved
        assert result["asr"]["model_type"] == "whisper"
        assert result["asr"]["model_name"] == "base"
        # User overrides applied
        assert result["device_preference"] == ["mps", "cpu"]
        assert result["asr"]["language"] == "auto"  # User override

    def test_empty_candidate_params(self):
        """Test when candidate has no params."""
        candidate_params = {}
        ui_config = {"asr": {"model_size": "small"}}

        result = _deep_merge(candidate_params, ui_config)

        assert result == {"asr": {"model_size": "small"}}

    def test_full_override_scenario(self):
        """Test complete override of nested config."""
        candidate_params = {
            "asr": {"model_type": "faster_whisper", "model_name": "large-v3"}
        }
        # Replace with completely different config
        ui_config = {
            "asr": {"model_type": "whisper", "model_name": "tiny"}
        }

        result = _deep_merge(candidate_params, ui_config)

        # Everything overridden
        assert result["asr"]["model_type"] == "whisper"
        assert result["asr"]["model_name"] == "tiny"

    def test_diarization_config_merge(self):
        """Test merging diarization config."""
        candidate_params = {
            "diarization": {"model_name": "pyannote_diarization"}
        }
        ui_config = {
            "diarization": {"max_speakers": 5}
        }

        result = _deep_merge(candidate_params, ui_config)

        assert result["diarization"]["model_name"] == "pyannote_diarization"
        assert result["diarization"]["max_speakers"] == 5
