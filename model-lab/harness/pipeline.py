"""
LCS-13: Pipeline Runner for Model Lab.

Enables chaining models in a linear pipeline:
- enhance → asr
- separate(vocals) → asr
- separate(vocals) → music_transcription

Interface:
- run_pipeline(pipeline_config, audio, sr) -> {artifacts, final}

Pipeline config schema:
  steps:
    - id: "enhance"
      model: "deepfilternet"
      surface: "enhance"
      args: {}
    - id: "asr"
      model: "moonshine"
      surface: "asr"
      args: {}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Configuration
# =============================================================================


@dataclass
class PipelineStep:
    """A single step in a pipeline."""
    
    id: str
    model: str
    surface: str
    args: dict[str, Any] = field(default_factory=dict)
    
    # For separate surface: which stem to extract
    stem: str | None = None
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineStep":
        """Create step from dict."""
        return cls(
            id=data["id"],
            model=data["model"],
            surface=data["surface"],
            args=data.get("args", {}),
            stem=data.get("stem"),
        )


@dataclass
class PipelineConfig:
    """Configuration for a pipeline."""
    
    name: str
    description: str
    steps: list[PipelineStep]
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        """Create config from dict."""
        return cls(
            name=data.get("name", "unnamed_pipeline"),
            description=data.get("description", ""),
            steps=[PipelineStep.from_dict(s) for s in data["steps"]],
        )
    
    @classmethod
    def from_yaml(cls, path: Path | str) -> "PipelineConfig":
        """Load config from YAML file."""
        import yaml
        
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


# =============================================================================
# Pipeline Result
# =============================================================================


@dataclass
class PipelineResult:
    """Result of running a pipeline."""
    
    artifacts: dict[str, Any]  # step_id -> output
    final: Any  # Last step's output
    steps_executed: list[str]
    sample_rate: int
    
    def get_artifact(self, step_id: str) -> Any:
        """Get artifact by step ID."""
        if step_id not in self.artifacts:
            raise KeyError(f"No artifact for step: {step_id}")
        return self.artifacts[step_id]


@dataclass
class PipelineError(Exception):
    """Error during pipeline execution."""
    
    step_id: str
    model_id: str
    surface: str
    original_error: Exception
    
    def __str__(self) -> str:
        return (
            f"Pipeline failed at step '{self.step_id}' "
            f"(model={self.model_id}, surface={self.surface}): "
            f"{self.original_error}"
        )


# =============================================================================
# Surface Handlers
# =============================================================================


def handle_enhance(
    bundle: dict[str, Any],
    audio: np.ndarray,
    sr: int,
    args: dict[str, Any],
    **kwargs,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """
    Handle enhance surface.
    
    Returns: (audio, sr, metadata)
    """
    enhance_fn = bundle["enhance"]["process"]
    result = enhance_fn(audio, sr=sr, **args)
    
    # Result may be just audio or a dict
    if isinstance(result, np.ndarray):
        return result, sr, {"enhanced": True}
    elif isinstance(result, dict):
        return result.get("audio", result), result.get("sr", sr), result
    else:
        return np.asarray(result), sr, {"enhanced": True}


def handle_separate(
    bundle: dict[str, Any],
    audio: np.ndarray,
    sr: int,
    args: dict[str, Any],
    stem: str | None = None,
    **kwargs,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """
    Handle separate surface.
    
    Extracts a specific stem (default: vocals).
    Returns: (stem_audio, sr, metadata)
    """
    separate_fn = bundle["separate"]["separate"]
    result = separate_fn(audio, sr=sr, **args)
    
    stem = stem or "vocals"  # Default to vocals
    stems = result.get("stems", {})
    output_sr = result.get("sr", sr)
    
    if stem not in stems:
        available = list(stems.keys())
        raise ValueError(f"Stem '{stem}' not found. Available: {available}")
    
    return stems[stem], output_sr, {"stem_extracted": stem, "all_stems": list(stems.keys())}


def handle_asr(
    bundle: dict[str, Any],
    audio: np.ndarray,
    sr: int,
    args: dict[str, Any],
    **kwargs,
) -> tuple[None, int, dict[str, Any]]:
    """
    Handle asr surface.
    
    Returns: (None, sr, asr_result)
    """
    transcribe_fn = bundle["asr"]["transcribe"]
    result = transcribe_fn(audio, sr=sr, **args)
    
    # ASR doesn't output audio, just text
    return None, sr, result


def handle_music_transcription(
    bundle: dict[str, Any],
    audio: np.ndarray,
    sr: int,
    args: dict[str, Any],
    **kwargs,
) -> tuple[None, int, dict[str, Any]]:
    """
    Handle music_transcription surface.
    
    Returns: (None, sr, transcription_result)
    """
    transcribe_fn = bundle["music_transcription"]["transcribe"]
    result = transcribe_fn(audio, sr=sr, **args)
    
    # Music transcription doesn't output audio
    return None, sr, result


def handle_classify(
    bundle: dict[str, Any],
    audio: np.ndarray,
    sr: int,
    args: dict[str, Any],
    **kwargs,
) -> tuple[None, int, dict[str, Any]]:
    """
    Handle classify surface.
    
    Returns: (None, sr, classification_result)
    """
    classify_fn = bundle["classify"]["predict"]
    result = classify_fn(audio, sr=sr, **args)
    
    return None, sr, result


def handle_embed(
    bundle: dict[str, Any],
    audio: np.ndarray,
    sr: int,
    args: dict[str, Any],
    **kwargs,
) -> tuple[None, int, dict[str, Any]]:
    """
    Handle embed surface.
    
    Returns: (None, sr, embedding_result)
    """
    encode_fn = bundle["embed"]["encode"]
    result = encode_fn(audio, sr=sr, **args)
    
    return None, sr, {"embedding": result}


# Surface -> handler mapping
SURFACE_HANDLERS: dict[str, Callable] = {
    "enhance": handle_enhance,
    "separate": handle_separate,
    "asr": handle_asr,
    "music_transcription": handle_music_transcription,
    "classify": handle_classify,
    "embed": handle_embed,
}


# =============================================================================
# Pipeline Runner
# =============================================================================


class PipelineRunner:
    """
    Runs pipelines of chained models.
    
    Example:
        runner = PipelineRunner()
        result = runner.run(config, audio, sr)
    """
    
    def __init__(
        self,
        model_loader: Callable[[str, dict], dict] | None = None,
        device: str = "cpu",
    ):
        """
        Initialize pipeline runner.
        
        Args:
            model_loader: Function to load models. If None, uses ModelRegistry.
            device: Default device for model loading.
        """
        self._model_loader = model_loader
        self._device = device
        self._loaded_models: dict[str, dict] = {}
    
    def _get_model_loader(self) -> Callable:
        """Get the model loader function."""
        if self._model_loader is not None:
            return self._model_loader
        
        # Import registry lazily
        from harness.registry import ModelRegistry
        
        def load(model_id: str, config: dict) -> dict:
            return ModelRegistry.load_model(model_id, config, device=self._device)
        
        return load
    
    def _load_model(self, model_id: str, config: dict) -> dict:
        """Load a model, caching for reuse."""
        cache_key = f"{model_id}:{hash(frozenset(config.items()) if config else 0)}"
        
        if cache_key not in self._loaded_models:
            loader = self._get_model_loader()
            self._loaded_models[cache_key] = loader(model_id, config)
            logger.info(f"Loaded model: {model_id}")
        
        return self._loaded_models[cache_key]
    
    def run(
        self,
        config: PipelineConfig | dict[str, Any],
        audio: np.ndarray,
        sr: int,
    ) -> PipelineResult:
        """
        Run a pipeline.
        
        Args:
            config: Pipeline configuration
            audio: Input audio
            sr: Sample rate
            
        Returns:
            PipelineResult with artifacts and final output
        """
        if isinstance(config, dict):
            config = PipelineConfig.from_dict(config)
        
        # Ensure numpy array
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        audio = np.asarray(audio, dtype=np.float32)
        
        artifacts: dict[str, Any] = {}
        steps_executed: list[str] = []
        current_audio = audio
        current_sr = sr
        final_output: Any = None
        
        logger.info(f"Starting pipeline: {config.name} ({len(config.steps)} steps)")
        
        for step in config.steps:
            logger.info(f"  Step: {step.id} ({step.model}/{step.surface})")
            
            try:
                # Get handler for this surface
                handler = SURFACE_HANDLERS.get(step.surface)
                if handler is None:
                    raise ValueError(f"Unknown surface: {step.surface}")
                
                # Load model
                bundle = self._load_model(step.model, step.args)
                
                # Verify surface is available
                if step.surface not in bundle:
                    available = [k for k in bundle if isinstance(bundle[k], dict)]
                    raise ValueError(
                        f"Surface '{step.surface}' not available in {step.model}. "
                        f"Available: {available}"
                    )
                
                # Run handler
                output_audio, output_sr, metadata = handler(
                    bundle=bundle,
                    audio=current_audio,
                    sr=current_sr,
                    args=step.args,
                    stem=step.stem,
                )
                
                # Store artifact
                artifacts[step.id] = {
                    "audio": output_audio,
                    "sr": output_sr,
                    "metadata": metadata,
                }
                
                # Update for next step
                if output_audio is not None:
                    current_audio = output_audio
                    current_sr = output_sr
                
                final_output = metadata
                steps_executed.append(step.id)
                
            except Exception as e:
                raise PipelineError(
                    step_id=step.id,
                    model_id=step.model,
                    surface=step.surface,
                    original_error=e,
                ) from e
        
        logger.info(f"Pipeline complete: {len(steps_executed)} steps executed")
        
        return PipelineResult(
            artifacts=artifacts,
            final=final_output,
            steps_executed=steps_executed,
            sample_rate=current_sr,
        )
    
    def clear_cache(self) -> None:
        """Clear loaded model cache."""
        self._loaded_models.clear()


# =============================================================================
# Convenience Functions
# =============================================================================


def run_pipeline(
    pipeline_config: PipelineConfig | dict[str, Any] | Path | str,
    audio: np.ndarray,
    sr: int,
    device: str = "cpu",
) -> PipelineResult:
    """
    Run a pipeline on audio.
    
    Args:
        pipeline_config: Config dict, PipelineConfig, or path to YAML
        audio: Input audio
        sr: Sample rate
        device: Device for model loading
        
    Returns:
        PipelineResult
    """
    if isinstance(pipeline_config, (str, Path)):
        pipeline_config = PipelineConfig.from_yaml(pipeline_config)
    
    runner = PipelineRunner(device=device)
    return runner.run(pipeline_config, audio, sr)


# =============================================================================
# Fake Models for Testing
# =============================================================================


class FakeEnhanceBundle:
    """Fake enhance bundle for testing."""
    
    @staticmethod
    def process(audio: np.ndarray, sr: int = 16000, **kwargs) -> np.ndarray:
        """Fake enhancement - returns audio with slight modification."""
        return audio * 0.99  # Slight attenuation to show processing


class FakeSeparateBundle:
    """Fake separate bundle for testing."""
    
    @staticmethod
    def separate(audio: np.ndarray, sr: int = 44100, **kwargs) -> dict:
        """Fake separation - splits audio into fake stems."""
        # Create fake stems from input
        return {
            "stems": {
                "vocals": audio * 0.6,
                "drums": audio * 0.2,
                "bass": audio * 0.15,
                "other": audio * 0.05,
            },
            "sr": sr,
        }


class FakeASRBundle:
    """Fake ASR bundle for testing."""
    
    @staticmethod
    def transcribe(audio: np.ndarray, sr: int = 16000, **kwargs) -> dict:
        """Fake transcription."""
        return {
            "text": "This is a fake transcription for testing.",
            "segments": [{"text": "This is a fake transcription for testing.", "start": 0, "end": 1}],
        }


class FakeMusicTranscriptionBundle:
    """Fake music transcription bundle for testing."""
    
    @staticmethod
    def transcribe(audio: np.ndarray, sr: int = 22050, **kwargs) -> dict:
        """Fake music transcription."""
        return {
            "notes": [
                {"onset": 0.0, "offset": 0.5, "pitch": 60, "velocity": 0.8},
                {"onset": 0.5, "offset": 1.0, "pitch": 64, "velocity": 0.7},
            ],
        }


def create_fake_bundle(surface: str) -> dict:
    """Create a fake bundle for a given surface."""
    bundles = {
        "enhance": {
            "model_type": "fake_enhance",
            "device": "cpu",
            "capabilities": ["enhance"],
            "modes": ["batch"],
            "enhance": {"process": FakeEnhanceBundle.process},
        },
        "separate": {
            "model_type": "fake_separate",
            "device": "cpu",
            "capabilities": ["separate"],
            "modes": ["batch"],
            "separate": {"separate": FakeSeparateBundle.separate},
        },
        "asr": {
            "model_type": "fake_asr",
            "device": "cpu",
            "capabilities": ["asr"],
            "modes": ["batch"],
            "asr": {"transcribe": FakeASRBundle.transcribe},
        },
        "music_transcription": {
            "model_type": "fake_music_transcription",
            "device": "cpu",
            "capabilities": ["music_transcription"],
            "modes": ["batch"],
            "music_transcription": {"transcribe": FakeMusicTranscriptionBundle.transcribe},
        },
    }
    
    if surface not in bundles:
        raise ValueError(f"No fake bundle for surface: {surface}")
    
    return bundles[surface]


def create_fake_model_loader() -> Callable:
    """Create a model loader that returns fake bundles."""
    def loader(model_id: str, config: dict) -> dict:
        # Map model_id to surface
        model_surfaces = {
            "fake_enhance": "enhance",
            "fake_separate": "separate",
            "fake_asr": "asr",
            "fake_music_transcription": "music_transcription",
        }
        
        surface = model_surfaces.get(model_id)
        if surface is None:
            raise ValueError(f"Unknown fake model: {model_id}")
        
        return create_fake_bundle(surface)
    
    return loader
