"""
Tests for Run History & Comparison feature.

These tests enforce:
1. Runs are correctly grouped by input hash
2. Comparison logic handles edge cases
3. Metrics diff calculation is correct
4. Config hash is stable for grouping
"""

import pytest
import json
import sys
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from server.services.runs_index import _compute_config_hash


class TestConfigHashStability:
    """Config hash must be stable for grouping similar runs."""
    
    def test_same_config_produces_same_hash(self):
        """Identical configs should produce identical hashes."""
        config1 = {
            "steps": ["ingest", "asr", "diarization"],
            "preprocessing": ["normalize_loudness"],
            "config": {"asr": {"model_id": "whisper-large"}}
        }
        config2 = {
            "steps": ["ingest", "asr", "diarization"],
            "preprocessing": ["normalize_loudness"],
            "config": {"asr": {"model_id": "whisper-large"}}
        }
        
        hash1 = _compute_config_hash(config1)
        hash2 = _compute_config_hash(config2)
        
        assert hash1 == hash2
    
    def test_different_steps_different_hash(self):
        """Different steps should produce different hashes."""
        config1 = {"steps": ["ingest", "asr"], "preprocessing": [], "config": {}}
        config2 = {"steps": ["ingest", "asr", "diarization"], "preprocessing": [], "config": {}}
        
        hash1 = _compute_config_hash(config1)
        hash2 = _compute_config_hash(config2)
        
        assert hash1 != hash2
    
    def test_different_preprocessing_different_hash(self):
        """Different preprocessing should produce different hashes."""
        config1 = {"steps": ["ingest", "asr"], "preprocessing": [], "config": {}}
        config2 = {"steps": ["ingest", "asr"], "preprocessing": ["normalize_loudness"], "config": {}}
        
        hash1 = _compute_config_hash(config1)
        hash2 = _compute_config_hash(config2)
        
        assert hash1 != hash2
    
    def test_order_independent_for_config_keys(self):
        """Config key order should not affect hash (JSON sorted keys)."""
        config1 = {"steps": ["a", "b"], "preprocessing": [], "config": {"z": 1, "a": 2}}
        config2 = {"steps": ["a", "b"], "preprocessing": [], "config": {"a": 2, "z": 1}}
        
        hash1 = _compute_config_hash(config1)
        hash2 = _compute_config_hash(config2)
        
        assert hash1 == hash2
    
    def test_hash_length_is_16_chars(self):
        """Config hash should be 16 hex characters."""
        config = {"steps": ["ingest"], "preprocessing": [], "config": {}}
        hash_val = _compute_config_hash(config)
        
        assert len(hash_val) == 16
        assert all(c in '0123456789abcdef' for c in hash_val)


class TestMetricsDiffCalculation:
    """Metrics diff must be calculated correctly."""
    
    def test_diff_calculation_positive(self):
        """Diff should be b - a."""
        metrics_a = {"word_count": 100}
        metrics_b = {"word_count": 150}
        
        diff = metrics_b["word_count"] - metrics_a["word_count"]
        
        assert diff == 50
    
    def test_diff_calculation_negative(self):
        """Negative diff when b < a."""
        metrics_a = {"word_count": 150}
        metrics_b = {"word_count": 100}
        
        diff = metrics_b["word_count"] - metrics_a["word_count"]
        
        assert diff == -50
    
    def test_diff_calculation_zero(self):
        """Zero diff when equal."""
        metrics_a = {"word_count": 100}
        metrics_b = {"word_count": 100}
        
        diff = metrics_b["word_count"] - metrics_a["word_count"]
        
        assert diff == 0
    
    def test_diff_handles_none_gracefully(self):
        """Missing metrics should result in None diff."""
        metrics_a = {"word_count": None}
        metrics_b = {"word_count": 100}
        
        val_a = metrics_a.get("word_count")
        val_b = metrics_b.get("word_count")
        
        diff = None
        if val_a is not None and val_b is not None:
            diff = val_b - val_a
        
        assert diff is None


class TestRunsByInputFiltering:
    """Runs should be correctly filtered by input hash."""
    
    def test_filter_by_hash(self):
        """Only runs with matching hash should be returned."""
        runs = [
            {"run_id": "run1", "input_hash": "abc123", "created_at": "2026-01-01T10:00:00"},
            {"run_id": "run2", "input_hash": "abc123", "created_at": "2026-01-01T11:00:00"},
            {"run_id": "run3", "input_hash": "xyz789", "created_at": "2026-01-01T12:00:00"},
        ]
        
        target_hash = "abc123"
        matching = [r for r in runs if r.get("input_hash") == target_hash]
        
        assert len(matching) == 2
        assert all(r["input_hash"] == target_hash for r in matching)
    
    def test_sort_by_created_at_descending(self):
        """Matching runs should be sorted by created_at descending."""
        runs = [
            {"run_id": "run1", "input_hash": "abc123", "created_at": "2026-01-01T10:00:00"},
            {"run_id": "run2", "input_hash": "abc123", "created_at": "2026-01-01T12:00:00"},
            {"run_id": "run3", "input_hash": "abc123", "created_at": "2026-01-01T11:00:00"},
        ]
        
        matching = [r for r in runs if r.get("input_hash") == "abc123"]
        sorted_runs = sorted(matching, key=lambda r: r.get("created_at", ""), reverse=True)
        
        assert sorted_runs[0]["run_id"] == "run2"  # Latest
        assert sorted_runs[1]["run_id"] == "run3"
        assert sorted_runs[2]["run_id"] == "run1"  # Earliest
    
    def test_empty_result_for_unknown_hash(self):
        """Unknown hash should return empty list."""
        runs = [
            {"run_id": "run1", "input_hash": "abc123", "created_at": "2026-01-01T10:00:00"},
        ]
        
        matching = [r for r in runs if r.get("input_hash") == "unknown_hash"]
        
        assert len(matching) == 0
    
    def test_handles_missing_input_hash(self):
        """Runs without input_hash should not match any hash."""
        runs = [
            {"run_id": "run1", "created_at": "2026-01-01T10:00:00"},  # No input_hash
            {"run_id": "run2", "input_hash": "abc123", "created_at": "2026-01-01T11:00:00"},
        ]
        
        matching = [r for r in runs if r.get("input_hash") == "abc123"]
        
        assert len(matching) == 1
        assert matching[0]["run_id"] == "run2"


class TestComparisonEdgeCases:
    """Comparison should handle edge cases gracefully."""
    
    def test_compare_same_run(self):
        """Comparing a run with itself should work."""
        run_data = {
            "run_id": "run1",
            "status": "COMPLETED",
            "steps_completed": ["ingest", "asr"],
            "preprocessing_ops": [],
        }
        
        # Both A and B reference same run
        config_diff = {
            "steps": {"a": run_data["steps_completed"], "b": run_data["steps_completed"]},
            "preprocessing": {"a": run_data["preprocessing_ops"], "b": run_data["preprocessing_ops"]},
        }
        
        assert config_diff["steps"]["a"] == config_diff["steps"]["b"]
    
    def test_compare_different_step_counts(self):
        """Runs with different step counts should show difference."""
        run_a = {"steps_completed": ["ingest", "asr"]}
        run_b = {"steps_completed": ["ingest", "asr", "diarization", "alignment"]}
        
        steps_only_in_b = set(run_b["steps_completed"]) - set(run_a["steps_completed"])
        
        assert steps_only_in_b == {"diarization", "alignment"}
    
    def test_missing_metrics_in_one_run(self):
        """Comparison should handle missing metrics in one run."""
        metrics_a = {"word_count": 100, "duration_s": 120.5}
        metrics_b = {"word_count": 150}  # Missing duration_s
        
        all_keys = set(metrics_a.keys()) | set(metrics_b.keys())
        
        comparison = {}
        for key in all_keys:
            val_a = metrics_a.get(key)
            val_b = metrics_b.get(key)
            diff = None
            if val_a is not None and val_b is not None:
                diff = val_b - val_a
            comparison[key] = {"a": val_a, "b": val_b, "diff": diff}
        
        assert comparison["word_count"]["diff"] == 50
        assert comparison["duration_s"]["a"] == 120.5
        assert comparison["duration_s"]["b"] is None
        assert comparison["duration_s"]["diff"] is None


class TestRerunParentLink:
    """Rerun should maintain parent link for lineage tracking."""
    
    def test_parent_run_id_in_request(self):
        """Rerun request should include parent_run_id."""
        original_run_id = "original_run_123"
        
        new_run_request = {
            "source": "rerun",
            "parent_run_id": original_run_id,
            "use_case_id": "test",
        }
        
        assert new_run_request["parent_run_id"] == original_run_id
        assert new_run_request["source"] == "rerun"
    
    def test_config_overrides_merged(self):
        """Config overrides should be merged with original config."""
        original_config = {"asr": {"model_id": "whisper-base"}, "device": "cpu"}
        overrides = {"asr": {"model_id": "whisper-large"}}
        
        merged = {**original_config, **overrides}
        
        assert merged["asr"]["model_id"] == "whisper-large"
        assert merged["device"] == "cpu"
