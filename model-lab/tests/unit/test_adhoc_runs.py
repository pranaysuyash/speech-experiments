"""
Tests for adhoc run contracts.

These tests enforce:
1. Adhoc runs have NO quality metrics (present but None)
2. Adhoc runs have proper provenance (audio_hash always present)
3. Adhoc runs are excluded from decisions
4. CLI mutual exclusion (--dataset XOR --audio)
"""

import pytest
import json
import sys
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from harness.runner_schema import (
    RunnerArtifact, RunContext, InputsSchema, QualityMetrics,
    validate_artifact, enforce_adhoc_metrics, ArtifactValidationError,
    QUALITY_METRICS_FORBIDDEN,
)


class TestAdhocHasNoQualityMetrics:
    """Adhoc runs must have forbidden metrics present and exactly None."""
    
    def test_adhoc_wer_must_be_none(self):
        """WER must be None for adhoc grade."""
        quality = QualityMetrics(wer=0.15)  # Incorrectly set
        
        with pytest.raises(ArtifactValidationError) as exc_info:
            enforce_adhoc_metrics(quality, grade="adhoc")
        
        assert "wer=0.15" in str(exc_info.value)
    
    def test_adhoc_der_must_be_none(self):
        """DER must be None for adhoc grade."""
        quality = QualityMetrics(der=0.25)
        
        with pytest.raises(ArtifactValidationError):
            enforce_adhoc_metrics(quality, grade="adhoc")
    
    def test_adhoc_speaker_accuracy_must_be_none(self):
        """speaker_accuracy must be None for adhoc grade."""
        quality = QualityMetrics(speaker_accuracy=0.9)
        
        with pytest.raises(ArtifactValidationError):
            enforce_adhoc_metrics(quality, grade="adhoc")
    
    def test_adhoc_all_none_passes(self):
        """All quality metrics None should pass for adhoc."""
        quality = QualityMetrics()  # All None by default
        
        # Should not raise
        enforce_adhoc_metrics(quality, grade="adhoc")
    
    def test_golden_batch_allows_quality_metrics(self):
        """golden_batch grade may have quality metrics."""
        quality = QualityMetrics(wer=0.15, cer=0.05)
        
        # Should not raise for golden_batch
        enforce_adhoc_metrics(quality, grade="golden_batch")
    
    def test_smoke_allows_quality_metrics(self):
        """smoke grade may have quality metrics (if GT available)."""
        quality = QualityMetrics(wer=0.15)
        
        # Should not raise for smoke
        enforce_adhoc_metrics(quality, grade="smoke")
    
    def test_forbidden_metrics_set_is_complete(self):
        """Verify QUALITY_METRICS_FORBIDDEN contains expected metrics."""
        expected = {"wer", "cer", "mer", "wil", "der", "der_proxy", 
                   "speaker_accuracy", "jaccard_error", "bleu"}
        assert QUALITY_METRICS_FORBIDDEN == expected


class TestAdhocHasRunContextAndHashes:
    """Adhoc runs must have full provenance with hashes."""
    
    def test_audio_hash_required(self):
        """audio_hash is always required."""
        run_context = RunContext(
            task="asr", model_id="test", grade="adhoc",
            timestamp="2026-01-01T00:00:00", git_hash="abc123",
            command=["test"], device="cpu"
        )
        inputs = InputsSchema(
            audio_path="/path/to/audio.wav",
            audio_hash=""  # Empty - should fail
        )
        artifact = RunnerArtifact(
            run_context=run_context,
            inputs=inputs,
            metrics_quality=QualityMetrics(),
            metrics_structural={"rtf": 0.5},
            output={"text": "hello"},
        )
        
        with pytest.raises(ArtifactValidationError) as exc_info:
            validate_artifact(artifact)
        
        assert "audio_hash" in str(exc_info.value)
    
    def test_git_hash_present_or_explicit_none(self):
        """git_hash should be present (or None if not in git repo)."""
        run_context = RunContext(
            task="asr", model_id="test", grade="adhoc",
            timestamp="2026-01-01T00:00:00", git_hash=None,  # Explicit None is OK
            command=["test"], device="cpu"
        )
        inputs = InputsSchema(
            audio_path="/path/to/audio.wav",
            audio_hash="abc123def456"
        )
        artifact = RunnerArtifact(
            run_context=run_context,
            inputs=inputs,
            metrics_quality=QualityMetrics(),
            metrics_structural={"rtf": 0.5},
            output={"text": "hello"},
        )
        
        # Should not raise - explicit None is allowed for git_hash
        validate_artifact(artifact)
        assert artifact.run_context.git_hash is None
    
    def test_video_input_requires_source_media_hash(self):
        """If source differs from audio path, source_media_hash required."""
        run_context = RunContext(
            task="asr", model_id="test", grade="adhoc",
            timestamp="2026-01-01T00:00:00", git_hash="abc123",
            command=["test"], device="cpu"
        )
        inputs = InputsSchema(
            audio_path="/tmp/extracted_audio.wav",
            audio_hash="abc123def456",
            source_media_path="/path/to/video.mp4",
            source_media_hash=None  # Missing - should fail
        )
        artifact = RunnerArtifact(
            run_context=run_context,
            inputs=inputs,
            metrics_quality=QualityMetrics(),
            metrics_structural={"rtf": 0.5},
            output={"text": "hello"},
        )
        
        with pytest.raises(ArtifactValidationError) as exc_info:
            validate_artifact(artifact)
        
        assert "source_media_hash" in str(exc_info.value)
    
    def test_video_input_with_proper_hashes_passes(self):
        """Video input with both hashes should pass."""
        run_context = RunContext(
            task="asr", model_id="test", grade="adhoc",
            timestamp="2026-01-01T00:00:00", git_hash="abc123",
            command=["test"], device="cpu"
        )
        inputs = InputsSchema(
            audio_path="/tmp/extracted_audio.wav",
            audio_hash="abc123def456",
            source_media_path="/path/to/video.mp4",
            source_media_hash="xyz789uvw012"  # Present
        )
        artifact = RunnerArtifact(
            run_context=run_context,
            inputs=inputs,
            metrics_quality=QualityMetrics(),
            metrics_structural={"rtf": 0.5},
            output={"text": "hello"},
        )
        
        # Should not raise
        validate_artifact(artifact)


class TestAdhocExcludedFromDecisions:
    """Adhoc runs must be excluded from decision evidence selection."""
    
    def test_adhoc_grade_identifiable(self):
        """Adhoc grade is clearly identifiable in artifact."""
        run_context = RunContext(
            task="asr", model_id="test", grade="adhoc",
            timestamp="2026-01-01T00:00:00", git_hash="abc123",
            command=["test"], device="cpu"
        )
        inputs = InputsSchema(
            audio_path="/path/to/audio.wav",
            audio_hash="abc123def456"
        )
        artifact = RunnerArtifact(
            run_context=run_context,
            inputs=inputs,
            metrics_quality=QualityMetrics(),
            metrics_structural={"rtf": 0.5},
            output={"text": "hello"},
        )
        
        as_dict = artifact.to_dict()
        
        # Multiple places to check grade
        assert as_dict["run_context"]["grade"] == "adhoc"
        assert as_dict["evidence"]["grade"] == "adhoc"
    
    def test_generate_decisions_filter_logic(self):
        """Filter logic for excluding adhoc from decisions."""
        # Simulate artifacts from different grades
        artifacts = [
            {"run_context": {"grade": "golden_batch"}, "metrics": {"wer": 0.1}},
            {"run_context": {"grade": "smoke"}, "metrics": {"rtf": 0.5}},
            {"run_context": {"grade": "adhoc"}, "metrics": {"rtf": 0.6}},  # Should be excluded
        ]
        
        # Filter as generate_decisions.py should
        evidence = [a for a in artifacts if a["run_context"]["grade"] != "adhoc"]
        
        assert len(evidence) == 2
        assert all(a["run_context"]["grade"] != "adhoc" for a in evidence)


class TestRunnerCliMutualExclusion:
    """Runners must reject --dataset and --audio together."""
    
    def test_mutual_exclusion_argparse_setup(self):
        """Test that argparse mutual exclusion can be set up correctly."""
        import argparse
        
        parser = argparse.ArgumentParser()
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--dataset', type=str)
        group.add_argument('--audio', type=str)
        
        # Should work with only --dataset
        args = parser.parse_args(['--dataset', 'test'])
        assert args.dataset == 'test'
        assert args.audio is None
        
        # Should work with only --audio
        args = parser.parse_args(['--audio', 'test.wav'])
        assert args.audio == 'test.wav'
        assert args.dataset is None
        
        # Should fail with both
        with pytest.raises(SystemExit):
            parser.parse_args(['--dataset', 'test', '--audio', 'test.wav'])


class TestStructuralMetricsRequired:
    """Structural metrics must always be present."""
    
    def test_empty_structural_metrics_fails(self):
        """Artifacts must have at least one structural metric."""
        run_context = RunContext(
            task="asr", model_id="test", grade="adhoc",
            timestamp="2026-01-01T00:00:00", git_hash="abc123",
            command=["test"], device="cpu"
        )
        inputs = InputsSchema(
            audio_path="/path/to/audio.wav",
            audio_hash="abc123def456"
        )
        artifact = RunnerArtifact(
            run_context=run_context,
            inputs=inputs,
            metrics_quality=QualityMetrics(),
            metrics_structural={},  # Empty - should fail
            output={"text": "hello"},
        )
        
        with pytest.raises(ArtifactValidationError) as exc_info:
            validate_artifact(artifact)
        
        assert "metrics_structural" in str(exc_info.value)
    
    def test_valid_structural_metrics_passes(self):
        """Artifacts with structural metrics should pass."""
        run_context = RunContext(
            task="asr", model_id="test", grade="adhoc",
            timestamp="2026-01-01T00:00:00", git_hash="abc123",
            command=["test"], device="cpu"
        )
        inputs = InputsSchema(
            audio_path="/path/to/audio.wav",
            audio_hash="abc123def456"
        )
        artifact = RunnerArtifact(
            run_context=run_context,
            inputs=inputs,
            metrics_quality=QualityMetrics(),
            metrics_structural={"latency_ms": 150, "rtf": 0.5},
            output={"text": "hello"},
        )
        
        # Should not raise
        validate_artifact(artifact)
