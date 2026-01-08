"""
Model registry for loading different models with consistent interface.
Supports LFM2.5-Audio, Whisper, SeamlessM4T, etc.
"""

import torch
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model lifecycle status."""
    EXPERIMENTAL = "experimental"
    CANDIDATE = "candidate"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


class ModelRegistry:
    """Central registry for model loading and configuration."""

    _models = {}
    _loaders = {}
    _metadata = {}  # Model metadata including status, version, etc.

    @classmethod
    def register_loader(cls,
                       model_type: str,
                       loader_func: Callable,
                       description: str = "",
                       status: ModelStatus = ModelStatus.EXPERIMENTAL,
                       version: str = "1.0.0",
                       performance_baseline: Optional[Dict[str, float]] = None):
        """
        Register a model loader function with metadata.

        Args:
            model_type: Model identifier (e.g., 'lfm2_5_audio', 'whisper')
            loader_func: Function that loads the model
            description: Human-readable description
            status: Model lifecycle status
            version: Model version string
            performance_baseline: Baseline performance metrics
        """
        cls._loaders[model_type] = {
            'loader': loader_func,
            'description': description
        }

        cls._metadata[model_type] = {
            'status': status.value,
            'version': version,
            'date_registered': datetime.now().isoformat(),
            'performance_baseline': performance_baseline or {},
            'hash': cls._calculate_model_hash(model_type, version)
        }

        logger.info(f"Registered loader: {model_type} v{version} ({status.value}) - {description}")

    @classmethod
    def _calculate_model_hash(cls, model_type: str, version: str) -> str:
        """Calculate a simple hash for model versioning."""
        import hashlib
        content = f"{model_type}:{version}:{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:8]

    @classmethod
    def get_model_metadata(cls, model_type: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a registered model."""
        return cls._metadata.get(model_type)

    @classmethod
    def update_model_status(cls, model_type: str, status: ModelStatus):
        """Update model status."""
        if model_type in cls._metadata:
            cls._metadata[model_type]['status'] = status.value
            cls._metadata[model_type]['date_updated'] = datetime.now().isoformat()
            logger.info(f"Updated {model_type} status to {status.value}")

    @classmethod
    def update_performance_baseline(cls, model_type: str, metrics: Dict[str, float]):
        """Update performance baseline for a model."""
        if model_type in cls._metadata:
            cls._metadata[model_type]['performance_baseline'].update(metrics)
            cls._metadata[model_type]['date_updated'] = datetime.now().isoformat()
            logger.info(f"Updated performance baseline for {model_type}")

    @classmethod
    def list_models_by_status(cls, status: ModelStatus) -> list:
        """List models by status."""
        return [model for model, meta in cls._metadata.items()
                if meta['status'] == status.value]

    @classmethod
    def get_production_models(cls) -> list:
        """Get list of production-ready models."""
        return cls.list_models_by_status(ModelStatus.PRODUCTION)

    @classmethod
    def validate_model_status(cls, model_type: str, required_status: ModelStatus = ModelStatus.PRODUCTION) -> bool:
        """Validate that model meets minimum status requirement."""
        metadata = cls.get_model_metadata(model_type)
        if not metadata:
            return False

        status_hierarchy = {
            ModelStatus.EXPERIMENTAL: 0,
            ModelStatus.CANDIDATE: 1,
            ModelStatus.PRODUCTION: 2,
            ModelStatus.DEPRECATED: -1
        }

        current_level = status_hierarchy.get(ModelStatus(metadata['status']), -1)
        required_level = status_hierarchy.get(required_status, 2)

        return current_level >= required_level

    @classmethod
    def load_model(cls,
                   model_type: str,
                   config: Dict[str, Any],
                   device: str = 'cpu') -> Any:
        """
        Load model using registered loader.

        Args:
            model_type: Model identifier
            config: Model configuration dictionary
            device: Target device

        Returns:
            Loaded model
        """
        if model_type not in cls._loaders:
            raise ValueError(f"No loader registered for: {model_type}")

        loader_info = cls._loaders[model_type]
        loader_func = loader_info['loader']

        logger.info(f"Loading {model_type} on {device}...")
        model = loader_func(config, device)
        logger.info(f"✓ Loaded {model_type}")

        cls._models[model_type] = model
        return model

    @classmethod
    def get_model(cls, model_type: str) -> Optional[Any]:
        """Get previously loaded model."""
        return cls._models.get(model_type)

    @classmethod
    def list_models(cls) -> list:
        """List registered model types."""
        return list(cls._loaders.keys())


# Model loader functions

def load_lfm2_5_audio(config: Dict[str, Any], device: str) -> Any:
    """
    Load LFM2.5-Audio model with device-aware processor initialization.
    
    Handles vendor library quirks:
    - liquid-audio's LFM2AudioProcessor.from_pretrained() defaults to device="cuda"
      even when CUDA is not available, causing crashes on MPS/CPU-only systems.
    - Workaround: Load processor on CPU first, then move to requested device.
    
    Args:
        config: Model configuration dictionary
        device: Target device ('cpu', 'mps', 'cuda', etc.)
    
    Returns:
        Dictionary with 'model', 'processor', 'device', and 'model_type' keys
        
    Note:
        The processor is loaded on CPU even when a different device is requested,
        because the liquid-audio library defaults to CUDA initialization.
        This is a defensive approach that works across all platforms.
        See: LFM_MPS_FIX_SUMMARY.md for detailed explanation.
    """
    try:
        from liquid_audio import LFM2AudioModel, LFM2AudioProcessor
        import torch

        model_name = config.get('model_name', 'LiquidAI/LFM2.5-Audio-1.5B')

        # Intelligent device selection: MPS > CPU (avoid CUDA due to vendor bug)
        if device == 'cuda':
            actual_device = 'cpu'
            logger.warning(f"LFM2.5-Audio: CUDA requested but using CPU due to vendor CUDA bug")
        elif device == 'mps':
            actual_device = 'mps'
            logger.info(f"LFM2.5-Audio: Using MPS (Apple Silicon) acceleration")
        else:
            actual_device = 'cpu'
            logger.info(f"LFM2.5-Audio: Using CPU")

        # Load model with selected device
        model = LFM2AudioModel.from_pretrained(model_name, device=actual_device).eval()

        # Use the official LFM processor
        # NOTE: liquid-audio's from_pretrained() defaults to device="cuda" and calls .to(device)
        # This fails on non-CUDA systems. Workaround: explicitly pass device="cpu" to from_pretrained(),
        # then move to the actual device if available
        try:
            processor = LFM2AudioProcessor.from_pretrained(model_name, device='cpu')
            logger.info(f"✓ LFM2AudioProcessor loaded successfully on CPU")
            
            # Move processor to the same device as model if not CPU
            if actual_device != 'cpu':
                try:
                    processor = processor.to(actual_device)
                    logger.info(f"✓ Processor moved to {actual_device}")
                except Exception:
                    # If move fails, keep on CPU - processor is lightweight
                    logger.warning(f"Could not move processor to {actual_device}, keeping on CPU")
            
        except Exception as e:
            logger.error(f"Failed to load LFM2AudioProcessor: {e}")
            raise RuntimeError(f"LFM2.5-Audio processor loading failed: {e}")

        return {
            'model': model,
            'processor': processor,
            'device': actual_device,
            'model_type': 'lfm2_5_audio'
        }

    except Exception as e:
        raise RuntimeError(f"LFM2.5-Audio loading failed: {e}")


def load_whisper(config: Dict[str, Any], device: str) -> Any:
    """Load Whisper model."""
    try:
        import whisper

        model_name = config.get('model_name', 'large-v3')
        model = whisper.load_model(model_name, device=device)

        return {
            'model': model,
            'device': device,
            'model_type': 'whisper'
        }

    except ImportError:
        raise ImportError("whisper package not installed. Install with: uv add openai-whisper")


def load_faster_whisper(config: Dict[str, Any], device: str) -> Any:
    """Load Faster-Whisper model."""
    try:
        from faster_whisper import WhisperModel

        model_name = config.get('model_name', 'large-v3')
        compute_type = config.get('inference', {}).get('compute_type', 'float16')

        # Map device names for faster-whisper
        if device == 'mps':
            device_fw = 'cpu'  # faster-whisper doesn't support MPS directly
        elif device == 'cuda':
            device_fw = 'cuda'
        else:
            device_fw = 'cpu'

        model = WhisperModel(
            model_name,
            device=device_fw,
            compute_type=compute_type
        )

        return {
            'model': model,
            'device': device,
            'device_fw': device_fw,  # Actual device used by faster-whisper
            'model_type': 'faster_whisper'
        }

    except ImportError:
        raise ImportError("faster-whisper package not installed. Install with: uv add faster-whisper")


def load_seamlessm4t(config: Dict[str, Any], device: str) -> Any:
    """Load SeamlessM4T model."""
    try:
        from transformers import SeamlessM4TForSpeechToText, AutoProcessor

        model_name = config.get('model_name', 'facebook/seamless-m4t-v2-large')
        processor = AutoProcessor.from_pretrained(model_name)
        model = SeamlessM4TForSpeechToText.from_pretrained(model_name)

        if device != 'cpu':
            model = model.to(device)

        model.eval()

        return {
            'model': model,
            'processor': processor,
            'device': device,
            'model_type': 'seamlessm4t'
        }

    except ImportError:
        raise ImportError("transformers package not installed. Install with: pip install transformers")


# Register default loaders
ModelRegistry.register_loader(
    'lfm2_5_audio',
    load_lfm2_5_audio,
    'LiquidAI LFM-2.5-Audio model for ASR, TTS, and conversation',
    status=ModelStatus.CANDIDATE,
    version="2.5.0",
    performance_baseline={'wer': 0.08, 'cer': 0.04}
)

ModelRegistry.register_loader(
    'whisper',
    load_whisper,
    'OpenAI Whisper model for ASR',
    status=ModelStatus.PRODUCTION,
    version="3.0.0",
    performance_baseline={'wer': 0.12, 'cer': 0.06}
)

ModelRegistry.register_loader(
    'faster_whisper',
    load_faster_whisper,
    'Optimized Whisper implementation for faster inference (guillaumekln/faster-whisper)',
    status=ModelStatus.PRODUCTION,
    version="1.0.0",
    performance_baseline={'wer': 0.12, 'cer': 0.06}
)

ModelRegistry.register_loader(
    'seamlessm4t',
    load_seamlessm4t,
    'Meta SeamlessM4T for multi-modal speech translation',
    status=ModelStatus.EXPERIMENTAL,
    version="2.0.0",
    performance_baseline={'wer': 0.15, 'cer': 0.08}
)


def load_model_from_config(config_path: Path, device: str = 'cpu') -> Any:
    """
    Load model from YAML config file.

    Args:
        config_path: Path to config.yaml
        device: Target device

    Returns:
        Loaded model
    """
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_type = config.get('model_type', 'unknown')
    return ModelRegistry.load_model(model_type, config, device)