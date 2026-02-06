"""Tests for new preprocessing operators."""

from harness.media_ingest import IngestConfig, build_ffmpeg_filter_chain
from harness.pipeline_config import (
    PREPROCESSING_REGISTRY,
    PipelineConfig,
    parse_preprocessing_op,
)


class TestPreprocessingRegistry:
    """Test that new operators exist in PREPROCESSING_REGISTRY."""

    def test_normalize_peak_in_registry(self):
        assert "normalize_peak" in PREPROCESSING_REGISTRY
        info = PREPROCESSING_REGISTRY["normalize_peak"]
        assert info["description"] == "Peak normalization"
        assert "target_db" in info["params"]
        assert info["params"]["target_db"]["default"] == -1.0

    def test_convert_samplerate_in_registry(self):
        assert "convert_samplerate" in PREPROCESSING_REGISTRY
        info = PREPROCESSING_REGISTRY["convert_samplerate"]
        assert info["description"] == "Explicit sample rate conversion"
        assert "target_sr" in info["params"]
        assert info["params"]["target_sr"]["default"] == 16000

    def test_mono_mix_in_registry(self):
        assert "mono_mix" in PREPROCESSING_REGISTRY
        info = PREPROCESSING_REGISTRY["mono_mix"]
        assert info["description"] == "Stereo to mono downmix"
        assert info["params"] == {}

    def test_compress_dynamics_in_registry(self):
        assert "compress_dynamics" in PREPROCESSING_REGISTRY
        info = PREPROCESSING_REGISTRY["compress_dynamics"]
        assert info["description"] == "Dynamic range compression"
        assert "threshold_db" in info["params"]
        assert "ratio" in info["params"]
        assert info["params"]["threshold_db"]["default"] == -20.0
        assert info["params"]["ratio"]["default"] == 4.0

    def test_gate_noise_in_registry(self):
        assert "gate_noise" in PREPROCESSING_REGISTRY
        info = PREPROCESSING_REGISTRY["gate_noise"]
        assert info["description"] == "Noise gate"
        assert "threshold_db" in info["params"]
        assert info["params"]["threshold_db"]["default"] == -40.0


class TestIngestConfigNewFields:
    """Test new fields in IngestConfig."""

    def test_peak_normalize_defaults(self):
        cfg = IngestConfig()
        assert cfg.peak_normalize is False
        assert cfg.peak_target_db == -1.0

    def test_compress_dynamics_defaults(self):
        cfg = IngestConfig()
        assert cfg.compress_dynamics is False
        assert cfg.compress_threshold_db == -20.0
        assert cfg.compress_ratio == 4.0

    def test_gate_noise_defaults(self):
        cfg = IngestConfig()
        assert cfg.gate_noise is False
        assert cfg.gate_threshold_db == -40.0

    def test_mono_mix_default(self):
        cfg = IngestConfig()
        assert cfg.mono_mix is False

    def test_custom_values(self):
        cfg = IngestConfig(
            peak_normalize=True,
            peak_target_db=-3.0,
            compress_dynamics=True,
            compress_threshold_db=-15.0,
            compress_ratio=6.0,
            gate_noise=True,
            gate_threshold_db=-50.0,
            mono_mix=True,
        )
        assert cfg.peak_normalize is True
        assert cfg.peak_target_db == -3.0
        assert cfg.compress_dynamics is True
        assert cfg.compress_threshold_db == -15.0
        assert cfg.compress_ratio == 6.0
        assert cfg.gate_noise is True
        assert cfg.gate_threshold_db == -50.0
        assert cfg.mono_mix is True


class TestFFmpegFilterChain:
    """Test ffmpeg filter chain generation for new operators."""

    def test_peak_normalize_filter(self):
        cfg = IngestConfig(peak_normalize=True, peak_target_db=-1.0)
        chain = build_ffmpeg_filter_chain(cfg)
        assert chain is not None
        assert "dynaudnorm" in chain

    def test_mono_mix_filter(self):
        cfg = IngestConfig(mono_mix=True)
        chain = build_ffmpeg_filter_chain(cfg)
        assert chain is not None
        assert "pan=mono|c0=0.5*c0+0.5*c1" in chain

    def test_compress_dynamics_filter(self):
        cfg = IngestConfig(compress_dynamics=True, compress_threshold_db=-20.0, compress_ratio=4.0)
        chain = build_ffmpeg_filter_chain(cfg)
        assert chain is not None
        assert "acompressor" in chain
        assert "threshold=-20.0dB" in chain
        assert "ratio=4.0" in chain

    def test_gate_noise_filter(self):
        cfg = IngestConfig(gate_noise=True, gate_threshold_db=-40.0)
        chain = build_ffmpeg_filter_chain(cfg)
        assert chain is not None
        assert "agate" in chain
        assert "threshold=-40.0dB" in chain

    def test_multiple_new_filters_combined(self):
        cfg = IngestConfig(
            peak_normalize=True,
            mono_mix=True,
            compress_dynamics=True,
            gate_noise=True,
        )
        chain = build_ffmpeg_filter_chain(cfg)
        assert chain is not None
        assert "dynaudnorm" in chain
        assert "pan=mono" in chain
        assert "acompressor" in chain
        assert "agate" in chain


class TestPipelineConfigToIngestConfig:
    """Test PipelineConfig.to_ingest_config() mapping for new operators."""

    def test_normalize_peak_mapping(self):
        config = PipelineConfig(
            steps=["ingest"],
            preprocessing=["normalize_peak(target_db=-3.0)"],
        )
        ingest_cfg = config.to_ingest_config()
        assert ingest_cfg.peak_normalize is True
        assert ingest_cfg.peak_target_db == -3.0

    def test_normalize_peak_default_params(self):
        config = PipelineConfig(
            steps=["ingest"],
            preprocessing=["normalize_peak"],
        )
        ingest_cfg = config.to_ingest_config()
        assert ingest_cfg.peak_normalize is True
        assert ingest_cfg.peak_target_db == -1.0

    def test_mono_mix_mapping(self):
        config = PipelineConfig(
            steps=["ingest"],
            preprocessing=["mono_mix"],
        )
        ingest_cfg = config.to_ingest_config()
        assert ingest_cfg.mono_mix is True

    def test_compress_dynamics_mapping(self):
        config = PipelineConfig(
            steps=["ingest"],
            preprocessing=["compress_dynamics(threshold_db=-15, ratio=6)"],
        )
        ingest_cfg = config.to_ingest_config()
        assert ingest_cfg.compress_dynamics is True
        assert ingest_cfg.compress_threshold_db == -15.0
        assert ingest_cfg.compress_ratio == 6.0

    def test_gate_noise_mapping(self):
        config = PipelineConfig(
            steps=["ingest"],
            preprocessing=["gate_noise(threshold_db=-50)"],
        )
        ingest_cfg = config.to_ingest_config()
        assert ingest_cfg.gate_noise is True
        assert ingest_cfg.gate_threshold_db == -50.0

    def test_convert_samplerate_mapping(self):
        config = PipelineConfig(
            steps=["ingest"],
            preprocessing=["convert_samplerate(target_sr=48000)"],
        )
        ingest_cfg = config.to_ingest_config()
        assert ingest_cfg.sample_rate == 48000

    def test_all_new_ops_combined(self):
        config = PipelineConfig(
            steps=["ingest"],
            preprocessing=[
                "normalize_peak(target_db=-2)",
                "mono_mix",
                "compress_dynamics(threshold_db=-18, ratio=5)",
                "gate_noise(threshold_db=-45)",
                "convert_samplerate(target_sr=22050)",
            ],
        )
        ingest_cfg = config.to_ingest_config()
        assert ingest_cfg.peak_normalize is True
        assert ingest_cfg.peak_target_db == -2.0
        assert ingest_cfg.mono_mix is True
        assert ingest_cfg.compress_dynamics is True
        assert ingest_cfg.compress_threshold_db == -18.0
        assert ingest_cfg.compress_ratio == 5.0
        assert ingest_cfg.gate_noise is True
        assert ingest_cfg.gate_threshold_db == -45.0
        assert ingest_cfg.sample_rate == 22050


class TestParsePreprocessingOp:
    """Test parsing of new operators."""

    def test_parse_normalize_peak_with_params(self):
        op_name, params = parse_preprocessing_op("normalize_peak(target_db=-3)")
        assert op_name == "normalize_peak"
        assert params == {"target_db": -3}

    def test_parse_compress_dynamics_with_params(self):
        op_name, params = parse_preprocessing_op("compress_dynamics(threshold_db=-20, ratio=4)")
        assert op_name == "compress_dynamics"
        assert params == {"threshold_db": -20, "ratio": 4}

    def test_parse_gate_noise_with_params(self):
        op_name, params = parse_preprocessing_op("gate_noise(threshold_db=-40)")
        assert op_name == "gate_noise"
        assert params == {"threshold_db": -40}

    def test_parse_mono_mix_no_params(self):
        op_name, params = parse_preprocessing_op("mono_mix")
        assert op_name == "mono_mix"
        assert params == {}

    def test_parse_convert_samplerate_with_params(self):
        op_name, params = parse_preprocessing_op("convert_samplerate(target_sr=48000)")
        assert op_name == "convert_samplerate"
        assert params == {"target_sr": 48000}
