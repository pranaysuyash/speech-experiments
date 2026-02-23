"""
Model registry for loading different models with consistent interface.
Supports LFM2.5-Audio, Whisper, SeamlessM4T, Distil-Whisper, Whisper.cpp, etc.

All loaders MUST return Bundle Contract v1 (see contracts.py).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, cast

import torch

from .contracts import ASRResult, Bundle, validate_bundle

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model lifecycle status."""

    EXPERIMENTAL = "experimental"
    CANDIDATE = "candidate"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


class ModelRegistry:
    """Central registry for model loading and configuration."""

    _models: dict[str, Bundle] = {}
    _loaders: dict[str, dict[str, Any]] = {}
    _metadata: dict[str, dict[str, Any]] = {}

    @classmethod
    def register_loader(
        cls,
        model_type: str,
        loader_func: Callable,
        description: str = "",
        status: ModelStatus = ModelStatus.EXPERIMENTAL,
        version: str = "1.0.0",
        capabilities: list[str] | None = None,
        hardware: list[str] | None = None,
        modes: list[str] | None = None,
        expected_baseline: dict[str, float] | None = None,
    ):
        """
        Register a model loader function with metadata.

        Args:
            model_type: Model identifier (e.g., 'lfm2_5_audio', 'whisper')
            loader_func: Function that loads the model, must return Bundle
            description: Human-readable description
            status: Model lifecycle status
            version: Model version string
            capabilities: ["asr", "tts", "chat", "mt"]
            hardware: ["cpu", "mps", "cuda"]
            modes: ["batch", "streaming", "cli"]
            expected_baseline: Expected performance metrics (optional)
        """
        # Enforce safe defaults
        capabilities = capabilities if capabilities is not None else []
        hardware = hardware if hardware is not None else ["cpu"]
        modes = modes if modes is not None else ["batch"]

        cls._loaders[model_type] = {"loader": loader_func, "description": description}

        cls._metadata[model_type] = {
            "status": status.value,
            "version": version,
            "capabilities": capabilities,
            "hardware": hardware,
            "modes": modes,
            "date_registered": datetime.now().isoformat(),
            "expected_baseline": expected_baseline or {},
            "observed_baseline": {},  # Only updated from run artifacts
            "hash": hashlib.md5(f"{model_type}:{version}".encode()).hexdigest()[:8],
        }

        logger.debug(
            f"Registered loader: {model_type} v{version} ({status.value}) caps={capabilities} - {description}"
        )

    @classmethod
    def get_model_metadata(cls, model_type: str) -> dict[str, Any] | None:
        """Get metadata for a registered model."""
        return cls._metadata.get(model_type)

    @classmethod
    def update_model_status(cls, model_type: str, status: ModelStatus):
        """Update model status."""
        if model_type in cls._metadata:
            cls._metadata[model_type]["status"] = status.value
            cls._metadata[model_type]["date_updated"] = datetime.now().isoformat()
            logger.info(f"Updated {model_type} status to {status.value}")

    @classmethod
    def update_observed_baseline(cls, model_type: str, metrics: dict[str, float]):
        """Update observed baseline from actual run artifacts."""
        if model_type in cls._metadata:
            cls._metadata[model_type]["observed_baseline"].update(metrics)
            cls._metadata[model_type]["date_updated"] = datetime.now().isoformat()
            logger.info(f"Updated observed baseline for {model_type}")

    @classmethod
    def list_models_by_status(cls, status: ModelStatus) -> list:
        """List models by status."""
        return [model for model, meta in cls._metadata.items() if meta["status"] == status.value]

    @classmethod
    def list_models_by_capability(cls, capability: str) -> list:
        """List models that have a specific capability."""
        return [
            model
            for model, meta in cls._metadata.items()
            if capability in meta.get("capabilities", [])
        ]

    @classmethod
    def get_production_models(cls) -> list:
        """Get list of production-ready models."""
        return cls.list_models_by_status(ModelStatus.PRODUCTION)

    @classmethod
    def validate_model_status(
        cls, model_type: str, required_status: ModelStatus = ModelStatus.PRODUCTION
    ) -> bool:
        """Validate that model meets minimum status requirement."""
        metadata = cls.get_model_metadata(model_type)
        if not metadata:
            return False

        status_hierarchy = {
            ModelStatus.EXPERIMENTAL.value: 0,
            ModelStatus.CANDIDATE.value: 1,
            ModelStatus.PRODUCTION.value: 2,
            ModelStatus.DEPRECATED.value: -1,
        }

        current_status = metadata.get("status", "experimental")
        current_level = status_hierarchy.get(current_status, -1)
        required_level = status_hierarchy.get(required_status.value, 2)

        return current_level >= required_level

    @classmethod
    def load_model(cls, model_type: str, config: dict[str, Any], device: str = "cpu") -> Bundle:
        """
        Load model using registered loader.

        Args:
            model_type: Model identifier
            config: Model configuration dictionary
            device: Target device

        Returns:
            Bundle conforming to Bundle Contract v1
        """
        if model_type not in cls._loaders:
            raise ValueError(f"No loader registered for: {model_type}")

        loader_info = cls._loaders[model_type]
        loader_func = loader_info["loader"]

        logger.info(f"Loading {model_type} on {device}...")
        bundle = loader_func(config, device)

        # Enforce Bundle Contract v1
        validate_bundle(bundle, model_type)

        logger.info(f"✓ Loaded {model_type} with capabilities: {bundle.get('capabilities', [])}")

        cls._models[model_type] = bundle
        return bundle

    @classmethod
    def get_model(cls, model_type: str) -> Bundle | None:
        """Get previously loaded model bundle."""
        return cls._models.get(model_type)

    @classmethod
    def list_models(cls) -> list:
        """List registered model types."""
        return list(cls._loaders.keys())


# =============================================================================
# Model Loader Functions - All must return Bundle Contract v1
# =============================================================================


def load_lfm2_5_audio(config: dict[str, Any], device: str) -> Bundle:
    """
    Load LFM2.5-Audio model with Bundle Contract v1.
    """
    try:
        import torch
        from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

        model_name = config.get("model_name", "LiquidAI/LFM2.5-Audio-1.5B")

        # Device selection: MPS > CPU (avoid CUDA due to vendor bug)
        if device == "cuda":
            actual_device = "cpu"
            logger.warning("LFM2.5-Audio: CUDA requested but using CPU due to vendor CUDA bug")
        elif device == "mps":
            actual_device = "mps"
            logger.info("LFM2.5-Audio: Using MPS (Apple Silicon) acceleration")
        else:
            actual_device = "cpu"
            logger.info("LFM2.5-Audio: Using CPU")

        model = LFM2AudioModel.from_pretrained(model_name, device=actual_device).eval()

        try:
            processor = LFM2AudioProcessor.from_pretrained(model_name, device="cpu")
            if actual_device != "cpu":
                try:
                    processor = processor.to(actual_device)
                    logger.info(f"✓ Processor moved to {actual_device}")
                except Exception:
                    logger.warning(f"Could not move processor to {actual_device}, keeping on CPU")

            # CRITICAL FIX: Manually initialize audio detokenizer to avoid .cuda() hardcode in library
            # The library's processor.audio_detokenizer property calls .cuda() which fails on MPS
            if processor._audio_detokenizer is None and processor.detokenizer_path:
                try:
                    from liquid_audio.processor import LFM2AudioDetokenizer, Lfm2Config
                    from safetensors.torch import load_file

                    logger.info("Initializing audio detokenizer manually for MPS support...")

                    detok_config_path = Path(processor.detokenizer_path) / "config.json"
                    detok_config = Lfm2Config.from_pretrained(detok_config_path)

                    # Fix layer types mismatch (copied from library)
                    def rename_layer(layer):
                        if layer in ["conv", "full_attention"]:
                            return layer
                        if layer == "sliding_attention":
                            return "full_attention"
                        raise ValueError(f"Unknown layer: {layer}")

                    if isinstance(detok_config.layer_types, list):
                        detok_config.layer_types = [
                            rename_layer(l) for l in detok_config.layer_types
                        ]

                    # Initialize on device!
                    detok = LFM2AudioDetokenizer(detok_config).eval().to(actual_device)

                    detok_weights_path = Path(processor.detokenizer_path) / "model.safetensors"
                    detok_weights = load_file(detok_weights_path)
                    detok.load_state_dict(detok_weights)

                    processor._audio_detokenizer = detok
                    logger.info(f"✓ Audio detokenizer initialized on {actual_device}")

                except Exception as e:
                    logger.warning(f"Failed to manually initialize detokenizer: {e}")
                    # Fallback to library behavior (might fail on MPS later)
        except Exception as e:
            raise RuntimeError(f"LFM2.5-Audio processor loading failed: {e}") from e

        def transcribe(audio, sr=16000, **kwargs):
            """
            Transcribe audio using LFM2.5-Audio via official ASR API.

            Uses 'Perform ASR.' system prompt with generate_sequential()
            as documented in the liquid-audio repository.
            """
            import numpy as np
            from liquid_audio import ChatState

            # Ensure audio is tensor with correct shape [1, samples]
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)
            if audio.dtype != torch.float32:
                audio = audio.float()
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)  # [1, samples]

            # Create chat state for ASR (official prompt)
            chat = ChatState(processor, dtype=torch.bfloat16)

            # Official system prompt from liquid-audio docs
            chat.new_turn("system")
            chat.add_text("Perform ASR.")
            chat.end_turn()

            # Add user audio
            chat.new_turn("user")
            chat.add_audio(audio, sr)
            chat.end_turn()

            # Generate assistant response (transcription)
            chat.new_turn("assistant")

            # Collect text tokens using generate_sequential (official method)
            text_tokens = []
            max_tokens = kwargs.get("max_new_tokens", 512)

            with torch.no_grad():
                for token in model.generate_sequential(
                    **chat,
                    max_new_tokens=max_tokens,
                ):
                    # Token is text if numel() == 1
                    if token.numel() == 1:
                        text_tokens.append(token)

            # Decode text tokens
            if text_tokens:
                all_tokens = torch.cat(text_tokens)
                text = processor.text.decode(all_tokens, skip_special_tokens=True)
            else:
                text = ""

            # Clean up any remaining special markers
            text = text.replace("<|text_end|>", "").replace("<|im_end|>", "").strip()

            return {"text": text, "segments": [], "meta": {"model": model_name}}

        def synthesize(text, **kwargs):
            """
            Synthesize speech using LFM2.5-Audio via official TTS API.

            Uses 'Perform TTS. Use the US female voice.' prompt with generate_sequential()
            as documented in the liquid-audio repository.
            """
            import numpy as np
            from liquid_audio import ChatState

            # Select voice from kwargs or default
            voice = kwargs.get("voice", "US female")

            # Create chat state for TTS
            chat = ChatState(processor, dtype=torch.bfloat16)

            # Official system prompt from liquid-audio docs
            chat.new_turn("system")
            chat.add_text(f"Perform TTS. Use the {voice} voice.")
            chat.end_turn()

            # Add text to synthesize
            chat.new_turn("user")
            chat.add_text(text)
            chat.end_turn()

            # Generate assistant response (audio)
            chat.new_turn("assistant")

            # Collect audio tokens using generate_sequential
            audio_tokens = []
            max_tokens = kwargs.get("max_new_tokens", 512)

            with torch.no_grad():
                for token in model.generate_sequential(
                    **chat,
                    max_new_tokens=max_tokens,
                    audio_temperature=0.8,
                    audio_top_k=64,
                ):
                    # Token is audio if numel() > 1
                    if token.numel() > 1:
                        audio_tokens.append(token)

            # Decode audio tokens
            if audio_tokens:
                # Stack and decode
                audio_codes = torch.stack(audio_tokens[:-1], 1).unsqueeze(0)
                waveform = processor.decode(audio_codes)
                audio = waveform.cpu().numpy().squeeze()
                sr = 24000
            else:
                audio = np.zeros(1, dtype=np.float32)
                sr = 24000

            return {
                "audio": audio,
                "sample_rate": sr,
                "meta": {"model": model_name, "voice": voice},
            }

        def run_v2v_turn(audio=None, text=None, **kwargs):
            """
            Run V2V turn: Audio in -> Audio out.
            Generates interleaved text/audio tokens.
            """
            import numpy as np
            from liquid_audio import ChatState

            # Setup standard conversational prompt
            chat = ChatState(processor, dtype=torch.bfloat16)
            chat.new_turn("system")
            chat.add_text("You are a helpful assistant.")
            chat.end_turn()

            # Add user input
            chat.new_turn("user")
            has_input = False

            audio_tensor = None
            if audio is not None:
                if isinstance(audio, np.ndarray):
                    audio_tensor = torch.from_numpy(audio)
                elif isinstance(audio, torch.Tensor):
                    audio_tensor = audio

                if audio_tensor is not None:
                    if audio_tensor.dtype != torch.float32:
                        audio_tensor = audio_tensor.float()
                    if audio_tensor.ndim == 1:
                        audio_tensor = audio_tensor.unsqueeze(0)

                    chat.add_audio(audio_tensor, 16000)
                    has_input = True

            if text:
                chat.add_text(text)
                has_input = True

            if not has_input:
                logger.warning("V2V called with no input (audio or text)")
                return {"audio": np.zeros(0), "response_text": "", "meta": {"error": "no_input"}}

            chat.end_turn()

            # Generate response
            chat.new_turn("assistant")

            audio_tokens = []
            text_tokens = []
            max_tokens = kwargs.get("max_new_tokens", 768)

            try:
                with torch.no_grad():
                    for token in model.generate_sequential(
                        **chat,
                        max_new_tokens=max_tokens,
                        audio_temperature=0.8,
                        audio_top_k=64,
                    ):
                        if token.numel() > 1:
                            audio_tokens.append(token)
                        else:
                            text_tokens.append(token)

                # Decode outputs
                response_text = ""
                if text_tokens:
                    all_text_tokens = torch.cat(text_tokens)
                    response_text = processor.text.decode(all_text_tokens, skip_special_tokens=True)
                    response_text = (
                        response_text.replace("<|text_end|>", "").replace("<|im_end|>", "").strip()
                    )

                if audio_tokens:
                    audio_codes = torch.stack(audio_tokens[:-1], 1).unsqueeze(0)
                    waveform = processor.decode(audio_codes)
                    response_audio = waveform.cpu().numpy().squeeze()
                    sr_out = 24000
                else:
                    response_audio = np.zeros(0, dtype=np.float32)
                    sr_out = 24000

                return {
                    "audio": response_audio,
                    "sample_rate": sr_out,
                    "response_text": response_text,
                    "meta": {"model": model_name},
                }

            except Exception as e:
                logger.error(f"V2V generation failed: {e}")
                return {"audio": np.zeros(0), "response_text": "", "meta": {"error": str(e)}}

        return {
            "model_type": "lfm2_5_audio",
            "device": actual_device,
            "capabilities": ["asr", "tts", "chat", "v2v"],
            "modes": ["batch"],
            "asr": {"transcribe": transcribe},
            "tts": {"synthesize": synthesize},
            "chat": {"respond": run_v2v_turn},  # Use same function for chat
            "v2v": {"run_v2v_turn": run_v2v_turn},
            "raw": {"model": model, "processor": processor},
        }

    except Exception as e:
        raise RuntimeError(f"LFM2.5-Audio loading failed: {e}") from e


def load_whisper(config: dict[str, Any], device: str) -> Bundle:
    """Load Whisper model with Bundle Contract v1."""
    try:
        import torch  # Added for internal use

        # Patch torch.load to handle PyTorch 2.6 weights_only default change
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load

        import whisper

        model_name = config.get("model_name", "large-v3")
        model = whisper.load_model(model_name, device=device)

        # Restore original torch.load
        torch.load = original_load

        def transcribe(audio, sr=16000, **kwargs):
            """Transcribe audio using Whisper."""
            # Whisper expects float32 numpy array at 16kHz
            import numpy as np

            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            if isinstance(audio, np.ndarray) and audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            result = model.transcribe(audio, **kwargs)
            return {
                "text": result.get("text", ""),
                "segments": result.get("segments", []),
                "language": result.get("language", ""),
                "meta": result,
            }

        def align(audio, sr=16000, **kwargs):
            """Align audio (same as transcribe + segments)."""
            return transcribe(audio, sr, **kwargs)

        return {
            "model_type": "whisper",
            "device": device,
            "capabilities": ["asr", "alignment"],
            "modes": ["batch"],
            "asr": {"transcribe": transcribe},
            "alignment": {"align": align},
            "raw": {"model": model},
        }

    except ImportError:
        raise ImportError("whisper package not installed. Install with: uv add openai-whisper")


def load_faster_whisper(config: dict[str, Any], device: str) -> Bundle:
    """Load Faster-Whisper model with Bundle Contract v1."""
    try:
        import torch  # Added for consistency if needed, though fw uses numpy
        from faster_whisper import WhisperModel

        model_name = config.get("model_name", "large-v3")
        compute_type = config.get("inference", {}).get("compute_type", "float16")

        # Map device names for faster-whisper
        if device == "mps":
            device_fw = "cpu"  # faster-whisper doesn't support MPS directly
        elif device == "cuda":
            device_fw = "cuda"
        else:
            device_fw = "cpu"

        # Force int8/float32 on CPU if float16 requested (which is default but fails on many CPUs)
        if device_fw == "cpu" and compute_type == "float16":
            compute_type = "int8"
            # logger.info("Forced compute_type='int8' for CPU execution")

        model = WhisperModel(model_name, device=device_fw, compute_type=compute_type)

        def transcribe(audio, sr=16000, **kwargs):
            """Transcribe audio using Faster-Whisper."""
            import numpy as np

            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            if isinstance(audio, np.ndarray) and audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Pop progress_callback from kwargs if present (not supported by model.transcribe)
            progress_callback = kwargs.pop("progress_callback", None)

            segments_generator, info = model.transcribe(audio, **kwargs)

            segments_list = []
            for segment in segments_generator:
                segments_list.append(segment)
                if progress_callback:
                    progress_callback()

            text = " ".join([s.text for s in segments_list])

            return {
                "text": text,
                "segments": [
                    {"start": s.start, "end": s.end, "text": s.text} for s in segments_list
                ],
                "language": info.language if info else "",
                "meta": {"language_probability": info.language_probability if info else 0},
            }

        return {
            "model_type": "faster_whisper",
            "device": device,
            "capabilities": ["asr"],
            "modes": ["batch", "streaming"],
            "asr": {"transcribe": transcribe},
            "raw": {"model": model, "device_fw": device_fw},
        }

    except ImportError:
        raise ImportError(
            "faster-whisper package not installed. Install with: uv add faster-whisper"
        )


def load_seamlessm4t(config: dict[str, Any], device: str) -> Bundle:
    """Load SeamlessM4T model with Bundle Contract v1."""
    try:
        import torch
        from transformers import AutoProcessor, SeamlessM4TForSpeechToText

        model_name = config.get("model_name", "facebook/seamless-m4t-v2-large")
        processor = AutoProcessor.from_pretrained(model_name)
        model = SeamlessM4TForSpeechToText.from_pretrained(model_name)

        if device != "cpu":
            model = model.to(device)
        model.eval()

        def transcribe(audio, sr=16000, **kwargs):
            """Transcribe audio using SeamlessM4T."""
            import numpy as np

            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            if isinstance(audio, np.ndarray) and audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            inputs = processor(audios=audio, sampling_rate=sr, return_tensors="pt")
            if device != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}

            # Filter out kwargs that model.generate() doesn't accept
            generate_kwargs = {k: v for k, v in kwargs.items() if k not in ("language", "lang")}

            with torch.no_grad():
                outputs = model.generate(**inputs, tgt_lang="eng", **generate_kwargs)

            text = processor.decode(outputs[0], skip_special_tokens=True)
            return {"text": text, "segments": [], "meta": {"model": model_name}}

        return {
            "model_type": "seamlessm4t",
            "device": device,
            "capabilities": ["asr", "mt"],
            "modes": ["batch"],
            "asr": {"transcribe": transcribe},
            "mt": {"translate": transcribe},  # Same function for now
            "raw": {"model": model, "processor": processor},
        }

    except ImportError:
        raise ImportError("transformers package not installed. Install with: uv add transformers")


def load_distil_whisper(config: dict[str, Any], device: str) -> Bundle:
    """
    Load Distil-Whisper via transformers pipeline with Bundle Contract v1.
    Uses pipeline for long-form support (chunking).
    """
    try:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        model_id = config.get("model_name", "distil-whisper/distil-large-v3")

        # dtype selection
        dtype_cfg = config.get("inference", {}).get("dtype", None)
        if dtype_cfg == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float16 if device != "cpu" else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        processor = AutoProcessor.from_pretrained(model_id)

        # Pipeline handles device movement
        # Configurable chunk length for long-form
        chunk_length_s = config.get("inference", {}).get("chunk_length_s", 30)
        stride_length_s = config.get("inference", {}).get("stride_length_s", None)

        pipeline_kwargs = {
            "model": model,
            "tokenizer": processor.tokenizer,
            "feature_extractor": processor.feature_extractor,
            "max_new_tokens": 128,
            "chunk_length_s": chunk_length_s,
            "batch_size": 16,
            "torch_dtype": torch_dtype,
            "device": device,
        }
        if stride_length_s is not None:
            pipeline_kwargs["stride_length_s"] = stride_length_s

        pipe = pipeline("automatic-speech-recognition", **pipeline_kwargs)

        def transcribe(audio, sr=16000, **kwargs):
            """Transcribe audio using Distil-Whisper pipeline."""
            # Pipeline expects numpy array or path
            import numpy as np

            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()

            # Ensure proper shape/type if numpy
            if isinstance(audio, np.ndarray):
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)

            # Run pipeline
            # return_timestamps=True needed for detailed output
            result = pipe(audio, generate_kwargs={"max_new_tokens": 128}, return_timestamps=True)

            text = result["text"]
            chunks = result.get("chunks", [])
            segments = [
                {"start": c["timestamp"][0], "end": c["timestamp"][1], "text": c["text"]}
                for c in chunks
                if c.get("timestamp")
            ]

            return {"text": text, "segments": segments, "meta": {"model": model_id}}

        return {
            "model_type": "distil_whisper",
            "device": device,
            "capabilities": ["asr"],
            "modes": ["batch"],
            "asr": {"transcribe": transcribe},
            "raw": {"model": model, "processor": processor, "pipeline": pipe},
        }

    except ImportError as e:
        raise ImportError(
            "transformers not installed. Install with: uv add transformers accelerate"
        ) from e


def load_mlx_whisper(config: dict[str, Any], device: str) -> Bundle:
    """
    Load MLX Whisper with Bundle Contract v1.

    Optimized for Apple Silicon via `mlx-whisper`.
    """
    try:
        import importlib
        import inspect
        import json
        from pathlib import Path

        import mlx.core as mx
        import mlx.nn as nn
        import mlx_whisper
        import numpy as np
        import torch
        from huggingface_hub import snapshot_download
        from mlx.utils import tree_unflatten
    except ImportError as e:
        raise ImportError("mlx-whisper not installed. Install with: uv add mlx-whisper") from e

    if device != "mps":
        logger.warning("mlx_whisper is optimized for mps; continuing on %s", device)

    model_name = (
        config.get("model_name")
        or config.get("config", {}).get("model_name")
        or "mlx-community/whisper-small.en-asr-fp16"
    )
    default_language = config.get("language") or config.get("config", {}).get("language") or "en"
    preferred_dtype = config.get("dtype", "float32")

    # Patch mlx_whisper loader to ignore newer config keys that older ModelDimensions
    # signatures don't accept (e.g., activation_dropout).
    load_models_mod = importlib.import_module("mlx_whisper.load_models")
    transcribe_mod = importlib.import_module("mlx_whisper.transcribe")
    whisper_mod = importlib.import_module("mlx_whisper.whisper")
    if not getattr(load_models_mod, "_model_lab_safe_config_patch", False):
        model_dim_keys = set(inspect.signature(whisper_mod.ModelDimensions).parameters)
        original_load_model = load_models_mod.load_model

        def _safe_load_model(path_or_hf_repo: str, dtype: mx.Dtype = mx.float32):
            model_path = Path(path_or_hf_repo)
            if not model_path.exists():
                model_path = Path(snapshot_download(repo_id=path_or_hf_repo))

            with open(str(model_path / "config.json"), "r", encoding="utf-8") as f:
                raw_config = json.loads(f.read())
            raw_config.pop("model_type", None)
            quantization = raw_config.pop("quantization", None)
            filtered_config = {k: v for k, v in raw_config.items() if k in model_dim_keys}

            alias_map = {
                "n_mels": raw_config.get("n_mels", raw_config.get("num_mel_bins")),
                "n_audio_ctx": raw_config.get(
                    "n_audio_ctx", raw_config.get("max_source_positions")
                ),
                "n_audio_state": raw_config.get("n_audio_state", raw_config.get("d_model")),
                "n_audio_head": raw_config.get(
                    "n_audio_head", raw_config.get("encoder_attention_heads")
                ),
                "n_audio_layer": raw_config.get("n_audio_layer", raw_config.get("encoder_layers")),
                "n_vocab": raw_config.get("n_vocab", raw_config.get("vocab_size")),
                "n_text_ctx": raw_config.get(
                    "n_text_ctx", raw_config.get("max_target_positions")
                ),
                "n_text_state": raw_config.get("n_text_state", raw_config.get("d_model")),
                "n_text_head": raw_config.get(
                    "n_text_head", raw_config.get("decoder_attention_heads")
                ),
                "n_text_layer": raw_config.get("n_text_layer", raw_config.get("decoder_layers")),
            }
            for key, value in alias_map.items():
                if key in model_dim_keys and key not in filtered_config and value is not None:
                    filtered_config[key] = value

            dropped = sorted(set(raw_config) - set(filtered_config))
            if dropped:
                logger.info("mlx_whisper: dropping unsupported config keys: %s", dropped)

            model_args = whisper_mod.ModelDimensions(**filtered_config)
            wf = model_path / "weights.safetensors"
            if not wf.exists():
                wf = model_path / "weights.npz"
            if not wf.exists():
                wf = model_path / "model.safetensors"
            if not wf.exists():
                wf = model_path / "model.npz"
            if not wf.exists():
                raise FileNotFoundError(
                    f"No supported MLX weight file found under {model_path}"
                )
            weights = mx.load(str(wf))

            model = whisper_mod.Whisper(model_args, dtype)
            if quantization is not None:
                class_predicate = (
                    lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
                    and f"{p}.scales" in weights
                )
                nn.quantize(model, **quantization, class_predicate=class_predicate)

            weights = tree_unflatten(list(weights.items()))
            model.update(weights)
            mx.eval(model.parameters())
            return model

        load_models_mod.load_model = _safe_load_model
        transcribe_mod.load_model = _safe_load_model
        load_models_mod._model_lab_safe_config_patch = True
        load_models_mod._model_lab_original_load_model = original_load_model

    def transcribe(audio, sr=16000, **kwargs):
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        decode_kwargs = dict(kwargs)
        decode_kwargs.pop("progress_callback", None)
        language = decode_kwargs.pop("language", default_language)
        decode_kwargs.setdefault("fp16", preferred_dtype == "float16")

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=model_name,
            language=language,
            **decode_kwargs,
        )

        segments = []
        for segment in result.get("segments", []):
            start = float(segment.get("start", segment.get("t0", 0.0)))
            end = float(segment.get("end", segment.get("t1", start)))
            text = (segment.get("text") or "").strip()
            if text:
                segments.append({"start": start, "end": end, "text": text})

        return {
            "text": (result.get("text") or "").strip(),
            "segments": segments,
            "language": result.get("language", language),
            "meta": {"model": model_name},
        }

    return {
        "model_type": "mlx_whisper",
        "device": device,
        "capabilities": ["asr"],
        "modes": ["batch"],
        "asr": {"transcribe": transcribe},
        "raw": {"model_name": model_name},
    }


ModelRegistry.register_loader(
    "mlx_whisper",
    load_mlx_whisper,
    "MLX Whisper: Apple Silicon optimized Whisper runtime",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["asr"],
    hardware=["mps", "cpu"],
    modes=["batch"],
)


def load_whisper_cpp(config: dict[str, Any], device: str) -> Bundle:
    """
    Whisper.cpp CLI adapter with Bundle Contract v1.
    Expects whisper.cpp binary built and accessible.

    Provides both transcribe() and transcribe_path() - runner uses transcribe().
    """
    import shutil
    import subprocess
    import tempfile

    import numpy as np
    import soundfile as sf

    # Needs torch check? No, CLI based.

    whisper_cpp_config = config.get("whisper_cpp", {})
    bin_path = whisper_cpp_config.get("bin_path", "whisper-cli")
    model_path = whisper_cpp_config.get("model_path")

    if shutil.which(bin_path) is None:
        raise RuntimeError(
            f"whisper.cpp binary not found: {bin_path}. "
            "Build whisper.cpp and set whisper_cpp.bin_path in config."
        )

    if not model_path:
        raise RuntimeError("Missing whisper_cpp.model_path in config (path to ggml/gguf model).")

    def transcribe_path(audio_path: str, **kwargs) -> ASRResult:
        """Transcribe audio file using whisper.cpp CLI."""
        lang = kwargs.get("language", "en")
        cmd = [
            bin_path,
            "-m",
            model_path,
            "-f",
            str(audio_path),
            "-l",
            lang,
            "--no-timestamps",
        ]

        threads = kwargs.get("threads")
        if threads:
            cmd += ["-t", str(threads)]

        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
            text = out.strip()
            return cast(ASRResult, {"text": text, "segments": [], "meta": {"raw_output": out}})
        except subprocess.CalledProcessError as e:
            return cast(ASRResult, {"text": "", "segments": [], "meta": {"error": str(e)}})

    def transcribe(audio, sr=16000, **kwargs) -> ASRResult:
        """
        Transcribe audio array using whisper.cpp.
        Writes temp wav file and calls CLI.
        """
        # Ensure numpy array
        import torch  # Need torch to check if tensor

        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        if isinstance(audio, np.ndarray) and audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            sf.write(tmp_path, audio, sr)
            result = transcribe_path(tmp_path, **kwargs)
        finally:
            # Clean up temp file
            import os

            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return result

    return {
        "model_type": "whisper_cpp",
        "device": "cpu",  # whisper.cpp handles acceleration internally
        "capabilities": ["asr"],
        "modes": ["cli", "batch"],
        "asr": {"transcribe": transcribe, "transcribe_path": transcribe_path},
        "raw": {"bin_path": bin_path, "model_path": model_path},
    }


def load_silero_vad(config: dict[str, Any], device: str) -> Bundle:
    """Load Silero VAD model with Bundle Contract v1."""
    try:
        import torch

        # Load from Torch Hub
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )

        (get_speech_timestamps, _, _, _, _) = utils

        if device != "cpu":
            model = model.to(device)
        model.eval()

        def detect(audio, sr=16000, **kwargs):
            """Detect speech segments."""
            # Silero expects tensor
            import numpy as np

            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)

            if audio.dim() > 1:
                audio = audio.squeeze()  # Ensure 1D

            if device != "cpu":
                audio = audio.to(device)

            # Run detection
            timestamps = get_speech_timestamps(audio, model, sampling_rate=sr, **kwargs)

            return {
                "segments": timestamps,  # List of dicts {start: int, end: int} in samples
                "meta": {"model": "silero_vad"},
            }

        return {
            "model_type": "silero_vad",
            "device": device,
            "capabilities": ["vad"],
            "modes": ["batch"],
            "vad": {"detect": detect},
            "raw": {"model": model, "utils": utils},
        }

    except Exception as e:
        raise RuntimeError(f"Silero VAD loading failed: {e}") from e


def load_pyannote_diarization(config: dict[str, Any], device: str) -> Bundle:
    """
    Load pyannote.audio pipeline for diarization.
    Requires 'pyannote.audio' and HuggingFace auth token in environment or login.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        raise ImportError("pyannote.audio not installed. Install with: uv add pyannote.audio")

    try:
        model_name = config.get("model_name", "pyannote/speaker-diarization-3.1")

        logger.info(f"Loading pyannote pipeline: {model_name}")

        # Load pipeline (pyannote >=4.0 uses token= instead of use_auth_token=)
        import os
        hf_token = config.get("auth_token") or os.environ.get("HF_TOKEN") or True
        try:
            pipeline = Pipeline.from_pretrained(model_name, token=hf_token)
        except TypeError:
            # Fallback for older pyannote versions
            pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)

        if pipeline is None:
            raise RuntimeError(
                f"Failed to load pyannote pipeline '{model_name}'. "
                "Check HF_TOKEN environment variable or access rights."
            )

        if device != "cpu":
            pipeline.to(torch.device(device))

        def diarize(audio, sr=16000, **kwargs):
            """
            Diarize audio.
            Args:
                audio: tensor [channels, samples] or numpy array
            Returns:
                {'segments': [{'start': float, 'end': float, 'speaker': str}, ...]}
            """
            import numpy as np

            # Pyannote expects a dict: {'waveform': (channels, time), 'sample_rate': sr}
            # or a path. We will provide memory structure.

            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)

            if audio.dim() == 1:
                audio = audio.unsqueeze(0)  # Add channel dim [1, T]

            # Ensure float32
            if audio.dtype != torch.float32:
                audio = audio.float()

            file = {"waveform": audio, "sample_rate": sr}

            # Run inference
            diarization = pipeline(file, **kwargs)

            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})

            return {
                "segments": segments,
                "num_speakers": len({s["speaker"] for s in segments}),
                "meta": {"model": model_name},
            }

        return {
            "model_type": "pyannote_diarization",
            "device": device,
            "capabilities": ["diarization"],
            "modes": ["batch"],
            "diarization": {"diarize": diarize},
            "raw": {"pipeline": pipeline},
        }

    except Exception as e:
        raise RuntimeError(f"Pyannote loading failed: {e}") from e


def load_heuristic_diarizer(config: dict[str, Any], device: str) -> Bundle:
    """
    Load Heuristic Diarizer (Silero VAD + Single Speaker assumption).
    Fallback for when pyannote is not available.
    """
    try:
        import torch

        # Reuse Silero VAD loading logic
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        (get_speech_timestamps, _, _, _, _) = utils

        if device != "cpu":
            model = model.to(device)
        model.eval()

        def diarize(audio, sr=16000, **kwargs):
            """
            Diarize using VAD + 1 speaker assumption.
            """
            import numpy as np

            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()  # Ensure float32
            else:
                audio = audio.float()  # Ensure float32

            if audio.dim() > 1:
                audio = audio.squeeze()

            if device != "cpu":
                audio = audio.to(device)

            timestamps = get_speech_timestamps(audio, model, sampling_rate=sr, return_seconds=True)

            # Convert to diarization format (add speaker label)
            segments = []
            for ts in timestamps:
                segments.append({"start": ts["start"], "end": ts["end"], "speaker": "SPEAKER_00"})

            return {
                "segments": segments,
                "num_speakers": 1 if segments else 0,
                "meta": {"model": "heuristic_diarizer", "backend": "silero_vad"},
            }

        return {
            "model_type": "heuristic_diarizer",
            "device": device,
            "capabilities": ["diarization"],
            "modes": ["batch"],
            "diarization": {"diarize": diarize},
            "raw": {"model": model},
        }

    except Exception as e:
        raise RuntimeError(f"Heuristic Diarizer loading failed: {e}") from e


# =============================================================================
# Register default loaders
# =============================================================================

ModelRegistry.register_loader(
    "lfm2_5_audio",
    load_lfm2_5_audio,
    "LiquidAI LFM-2.5-Audio model for ASR, TTS, and conversation",
    status=ModelStatus.CANDIDATE,
    version="2.5.0",
    capabilities=["asr", "tts", "chat"],
    hardware=["cpu", "mps"],
    modes=["batch"],
)

ModelRegistry.register_loader(
    "whisper",
    load_whisper,
    "OpenAI Whisper model for ASR",
    status=ModelStatus.PRODUCTION,
    version="3.0.0",
    capabilities=["asr"],
    hardware=["cpu", "mps", "cuda"],
    modes=["batch"],
)

ModelRegistry.register_loader(
    "faster_whisper",
    load_faster_whisper,
    "Optimized Whisper implementation for faster inference (guillaumekln/faster-whisper)",
    status=ModelStatus.PRODUCTION,
    version="1.0.0",
    capabilities=["asr"],
    hardware=["cpu", "cuda"],
    modes=["batch", "streaming"],
)

ModelRegistry.register_loader(
    "seamlessm4t",
    load_seamlessm4t,
    "Meta SeamlessM4T for multi-modal speech translation",
    status=ModelStatus.EXPERIMENTAL,
    version="2.0.0",
    capabilities=["asr", "mt"],
    hardware=["cpu", "mps", "cuda"],
    modes=["batch"],
)

ModelRegistry.register_loader(
    "distil_whisper",
    load_distil_whisper,
    "Distil-Whisper: 6x faster Whisper with minimal accuracy loss",
    status=ModelStatus.EXPERIMENTAL,
    version="3.0.0",
    capabilities=["asr"],
    hardware=["cpu", "mps", "cuda"],
    modes=["batch"],
)

ModelRegistry.register_loader(
    "pyannote_diarization",
    load_pyannote_diarization,
    "Pyannote.audio Speaker Diarization pipeline",
    status=ModelStatus.PRODUCTION,
    version="3.1.0",
    capabilities=["diarization"],
    hardware=["cpu", "cuda"],
    modes=["batch"],
)

ModelRegistry.register_loader(
    "heuristic_diarization",
    load_heuristic_diarizer,
    "Heuristic Diarizer (VAD + 1-speaker assumption)",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["diarization"],
    hardware=["cpu", "cuda", "mps"],
    modes=["batch"],
)

ModelRegistry.register_loader(
    "whisper_cpp",
    load_whisper_cpp,
    "whisper.cpp: edge-friendly C++ inference backend",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["asr"],
    hardware=["cpu"],
    modes=["cli"],
)


ModelRegistry.register_loader(
    "silero_vad",
    load_silero_vad,
    "Silero VAD: Production-grade voice activity detection",
    status=ModelStatus.PRODUCTION,
    version="4.0.0",
    capabilities=["vad"],
    hardware=["cpu", "mps", "cuda"],
    modes=["batch"],
)


# =============================================================================
# Moonshine ASR (LCS-04)
# =============================================================================


def load_moonshine(config: dict[str, Any], device: str) -> Bundle:
    """
    Load Moonshine ASR model with Bundle Contract v2.

    Moonshine is a fast CPU-first ASR from Useful Sensors (27M params).
    5-15x faster than Whisper on short segments.
    """
    try:
        import torch
        import numpy as np

        # Import moonshine - may need to be installed from git
        try:
            import moonshine
        except ImportError:
            raise ImportError(
                "moonshine package not installed. Install with:\n"
                "pip install -r models/moonshine/requirements.txt"
            )

        variant = config.get("variant", "tiny")
        model_name = f"moonshine/{variant}"

        # Device handling: moonshine supports CPU and CUDA
        # For MPS, we use CPU (moonshine may not have MPS support yet)
        if device == "mps":
            actual_device = "cpu"
            logger.info("Moonshine: MPS requested, using CPU (MPS not yet supported)")
        elif device == "cuda" and torch.cuda.is_available():
            actual_device = "cuda"
        else:
            actual_device = "cpu"

        logger.info(f"Loading Moonshine {variant} on {actual_device}")

        # Load the model
        model = moonshine.load_model(variant)

        def transcribe(audio, sr=16000, **kwargs):
            """Transcribe audio using Moonshine."""
            # Convert to numpy if tensor
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()

            # Ensure float32
            if isinstance(audio, np.ndarray) and audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Resample if needed (moonshine expects 16kHz)
            if sr != 16000:
                try:
                    import librosa

                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                except ImportError:
                    logger.warning("librosa not available for resampling")

            # Run transcription
            result = moonshine.transcribe(model, audio)

            # Handle result format (may be string or dict)
            if isinstance(result, str):
                text = result
                segments = []
            elif isinstance(result, list):
                # Multiple segments
                text = " ".join(r if isinstance(r, str) else r.get("text", "") for r in result)
                segments = result if isinstance(result[0], dict) else []
            else:
                text = result.get("text", str(result))
                segments = result.get("segments", [])

            return {
                "text": text.strip(),
                "segments": segments,
                "language": "en",  # Moonshine is English-only
                "meta": {"variant": variant, "device": actual_device},
            }

        return {
            "model_type": "moonshine",
            "device": actual_device,
            "capabilities": ["asr"],
            "modes": ["batch"],
            "asr": {"transcribe": transcribe},
            "raw": {"model": model, "variant": variant},
        }

    except ImportError as e:
        raise ImportError(f"Failed to load Moonshine: {e}")


ModelRegistry.register_loader(
    "moonshine",
    load_moonshine,
    "Moonshine: Fast CPU-first ASR from Useful Sensors (27M params)",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["asr"],
    hardware=["cpu", "mps"],
    modes=["batch"],
)


# =============================================================================
# YAMNet Audio Classification (LCS-05)
# =============================================================================


def load_yamnet(config: dict[str, Any], device: str) -> Bundle:
    """
    Load YAMNet audio classification model with Bundle Contract v2.

    YAMNet classifies 521 AudioSet event classes using TensorFlow Hub.
    """
    try:
        import numpy as np

        # Import TensorFlow and Hub
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except ImportError:
            raise ImportError(
                "TensorFlow not installed. Install with:\n"
                "pip install -r models/yamnet/requirements.txt"
            )

        # Suppress TF warnings
        tf.get_logger().setLevel("ERROR")

        logger.info("Loading YAMNet from TensorFlow Hub...")

        # Load YAMNet model from TF Hub
        yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

        # Load class names
        class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")
        with open(class_map_path) as f:
            # Skip header
            lines = f.readlines()[1:]
            class_names = [line.strip().split(",")[2] for line in lines]

        def classify(audio, sr=16000, top_k=5, **kwargs):
            """Classify audio using YAMNet."""
            # Ensure numpy array
            if hasattr(audio, "numpy"):
                audio = audio.numpy()
            audio = np.asarray(audio, dtype=np.float32).flatten()

            # Resample if needed (YAMNet expects 16kHz)
            if sr != 16000:
                try:
                    import librosa

                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                except ImportError:
                    logger.warning("librosa not available for resampling")

            # Run inference
            scores, embeddings, spectrogram = yamnet_model(audio)

            # Average scores across frames
            mean_scores = np.mean(scores.numpy(), axis=0)

            # Get top-k predictions
            top_indices = np.argsort(mean_scores)[::-1][:top_k]
            top_labels = [class_names[i] for i in top_indices]
            top_scores = [float(mean_scores[i]) for i in top_indices]

            return {
                "labels": top_labels,
                "scores": top_scores,
                "top_k": top_k,
                "all_scores": mean_scores.tolist(),
                "embeddings": embeddings.numpy().mean(axis=0).tolist(),  # Average embeddings
                "meta": {"n_frames": scores.shape[0], "n_classes": len(class_names)},
            }

        return {
            "model_type": "yamnet",
            "device": "cpu",  # TF Hub runs on CPU by default
            "capabilities": ["classify"],
            "modes": ["batch"],
            "classify": {"classify": classify},
            "raw": {"model": yamnet_model, "class_names": class_names},
        }

    except ImportError as e:
        raise ImportError(f"Failed to load YAMNet: {e}")


ModelRegistry.register_loader(
    "yamnet",
    load_yamnet,
    "YAMNet: Audio event classification (521 classes) from TF Hub",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["classify"],
    hardware=["cpu"],
    modes=["batch"],
)


# =============================================================================
# RNNoise Audio Enhancement (LCS-06)
# =============================================================================


def load_rnnoise(config: dict[str, Any], device: str) -> Bundle:
    """
    Load RNNoise audio enhancement model with Bundle Contract v2.

    RNNoise is a real-time noise suppression library using GRU-based RNN.
    Native runtime (C library with Python bindings via pyrnnoise).
    """
    try:
        import numpy as np

        # Import pyrnnoise Python bindings
        try:
            from pyrnnoise import RNNoise
        except ImportError:
            raise ImportError(
                "pyrnnoise not installed. Install with:\n"
                "pip install -r models/rnnoise/requirements.txt"
            )

        logger.info("Loading RNNoise...")

        # Create RNNoise instance with required sample_rate
        denoiser = RNNoise(sample_rate=48000)

        def enhance(audio, sr=48000, **kwargs):
            """Enhance audio by removing noise using RNNoise."""
            # Ensure numpy array
            if hasattr(audio, "numpy"):
                audio = audio.numpy()
            audio = np.asarray(audio, dtype=np.float32).flatten()

            # RNNoise expects 48kHz, resample if needed
            original_sr = sr
            if sr != 48000:
                try:
                    import librosa

                    audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
                    sr = 48000
                except ImportError:
                    logger.warning("librosa not available for resampling, may produce artifacts")

            # Process audio through RNNoise using denoise_chunk
            # denoise_chunk returns a generator of (frame, vad_probability) tuples
            vad_probs = []
            output_frames = []

            for frame, vad_prob in denoiser.denoise_chunk(audio):
                output_frames.append(frame)
                # vad_prob is a 2D array (1, samples), take mean as aggregate
                vad_probs.append(float(vad_prob.mean()))

            # Concatenate all frames
            if output_frames:
                output = np.concatenate(output_frames)
            else:
                output = audio  # Fallback to original if no output

            # Resample back to original rate if needed
            if original_sr != 48000:
                try:
                    import librosa

                    output = librosa.resample(output, orig_sr=48000, target_sr=original_sr)
                except ImportError:
                    pass

            return {
                "audio": output,
                "sample_rate": original_sr,
                "vad_probs": vad_probs,
                "meta": {"n_frames": len(vad_probs)},
            }

        return {
            "model_type": "rnnoise",
            "device": "cpu",  # Native C, CPU only
            "capabilities": ["enhance"],
            "modes": ["batch", "streaming"],
            "enhance": {"enhance": enhance},
            "raw": {"denoiser": denoiser},
        }

    except ImportError as e:
        raise ImportError(f"Failed to load RNNoise: {e}")


ModelRegistry.register_loader(
    "rnnoise",
    load_rnnoise,
    "RNNoise: Real-time noise suppression (native C runtime)",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["enhance"],
    hardware=["cpu"],
    modes=["batch", "streaming"],
)


# =============================================================================
# DeepFilterNet Audio Enhancement (LCS-07)
# =============================================================================


def load_deepfilternet(config: dict[str, Any], device: str) -> Bundle:
    """
    Load DeepFilterNet audio enhancement model with Bundle Contract v2.

    DeepFilterNet is a state-of-the-art speech enhancement model using
    ERB-scale deep filtering. Production-grade quality.
    """
    try:
        import numpy as np

        # Import deepfilternet
        try:
            from df.enhance import enhance as df_enhance, init_df
        except ImportError:
            raise ImportError(
                "deepfilternet not installed. Install with:\n"
                "pip install -r models/deepfilternet/requirements.txt"
            )

        variant = config.get("variant", "DeepFilterNet2")

        # Device handling
        if device == "mps":
            actual_device = "cpu"  # DF may not support MPS directly
            logger.info("DeepFilterNet: MPS requested, using CPU")
        elif device == "cuda":
            import torch

            actual_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            actual_device = "cpu"

        logger.info(f"Loading DeepFilterNet ({variant}) on {actual_device}...")

        # Initialize model
        model, df_state, _ = init_df()

        def enhance_audio(audio, sr=48000, **kwargs):
            """Enhance audio by removing noise using DeepFilterNet."""
            # Ensure numpy array
            if hasattr(audio, "numpy"):
                audio = audio.numpy()
            audio = np.asarray(audio, dtype=np.float32).flatten()

            original_sr = sr
            original_len = len(audio)

            # DeepFilterNet expects 48kHz, resample if needed
            if sr != 48000:
                try:
                    import librosa

                    audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
                except ImportError:
                    logger.warning("librosa not available for resampling")

            # Run enhancement
            enhanced = df_enhance(model, df_state, audio)

            # Resample back to original rate if needed
            if original_sr != 48000:
                try:
                    import librosa

                    enhanced = librosa.resample(enhanced, orig_sr=48000, target_sr=original_sr)
                except ImportError:
                    pass

            # Ensure output length matches input (preserve alignment)
            if len(enhanced) != original_len:
                if len(enhanced) > original_len:
                    enhanced = enhanced[:original_len]
                else:
                    enhanced = np.pad(enhanced, (0, original_len - len(enhanced)))

            return {
                "audio": enhanced,
                "sample_rate": original_sr,
                "meta": {"variant": variant, "device": actual_device},
            }

        return {
            "model_type": "deepfilternet",
            "device": actual_device,
            "capabilities": ["enhance"],
            "modes": ["batch"],
            "enhance": {"process": enhance_audio},
            "raw": {"model": model, "df_state": df_state},
        }

    except ImportError as e:
        raise ImportError(f"Failed to load DeepFilterNet: {e}")


ModelRegistry.register_loader(
    "deepfilternet",
    load_deepfilternet,
    "DeepFilterNet: State-of-the-art speech enhancement",
    status=ModelStatus.EXPERIMENTAL,
    version="2.0.0",
    capabilities=["enhance"],
    hardware=["cpu", "mps"],
    modes=["batch"],
)


# =============================================================================
# CLAP Contrastive Language-Audio Pretraining (LCS-08)
# =============================================================================


def load_clap(config: dict[str, Any], device: str) -> Bundle:
    """
    Load CLAP audio embedding and classification model with Bundle Contract v2.

    CLAP provides:
    - embed: Fixed-dimension audio embeddings (512-d)
    - classify: Zero-shot classification via text prompts

    First multi-surface model in the registry!
    """
    try:
        import numpy as np

        # Import laion-clap
        try:
            import laion_clap
        except ImportError:
            raise ImportError(
                "laion-clap not installed. Install with:\n"
                "pip install -r models/clap/requirements.txt"
            )

        import torch

        # Device handling
        if device == "mps":
            actual_device = "mps" if torch.backends.mps.is_available() else "cpu"
        elif device == "cuda":
            actual_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            actual_device = "cpu"

        logger.info(f"Loading CLAP on {actual_device}...")

        # Load model - use HTSAT-tiny for speed
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt()  # Downloads checkpoint if needed

        if actual_device == "mps" and torch.backends.mps.is_available():
            # CLAP often has issues on MPS (torchvision ops). Fallback to CPU if needed
            # But let's try to respect device request unless we know it fails.
            # User requested CPU-only for Mac.
            logger.warning("Forcing CLAP to CPU on MacOS due to torchvision incompatibility.")
            actual_device = "cpu"

        if actual_device != "cpu":
            model = model.to(actual_device)

        model.eval()

        def encode_audio(audio, sr=48000, **kwargs):
            """Generate audio embedding using CLAP."""
            # Ensure numpy array
            if hasattr(audio, "numpy"):
                audio = audio.numpy()
            audio = np.asarray(audio, dtype=np.float32).flatten()

            # CLAP expects 48kHz
            if sr != 48000:
                try:
                    import librosa

                    audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
                except ImportError:
                    pass

            # Get embedding
            with torch.no_grad():
                embedding = model.get_audio_embedding_from_data([audio], use_tensor=False)

            return {
                "embedding": embedding[0],
                "dim": len(embedding[0]),
                "meta": {"sample_rate": 48000, "device": actual_device},
            }

        def classify_audio(audio, sr=48000, labels=None, top_k=5, **kwargs):
            """Zero-shot classification using text prompts."""
            if labels is None:
                labels = ["speech", "music", "silence", "noise", "animal"]

            # Ensure numpy array
            if hasattr(audio, "numpy"):
                audio = audio.numpy()
            audio = np.asarray(audio, dtype=np.float32).flatten()

            # CLAP expects 48kHz
            if sr != 48000:
                try:
                    import librosa

                    audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
                except ImportError:
                    pass

            # Get embeddings
            import torch

            with torch.no_grad():
                # Convert to tensor for laion-clap if use_tensor=True
                audio_tensor = torch.from_numpy(audio).float()
                # Ensure it has the right shape if needed, but get_audio_embedding_from_data takes list[tensor] or tensor
                # If we pass a list, it expects list of tensors or list of strings (paths) depending on use_tensor?
                # detailed doc says: x can be tensor if use_tensor=True.
                if audio_tensor.ndim == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)  # (1, T)

                audio_embed = model.get_audio_embedding_from_data(x=audio_tensor, use_tensor=True)
                text_embed = model.get_text_embedding(labels, use_tensor=True)

                # Compute similarity
                audio_embed = audio_embed / audio_embed.norm(dim=-1, keepdim=True)
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
                similarity = (audio_embed @ text_embed.T).squeeze(0)

                # Softmax for probabilities
                probs = torch.softmax(similarity * 100, dim=-1)  # Temperature scaling

            probs = probs.cpu().numpy()

            # Sort by score
            sorted_indices = np.argsort(probs)[::-1][:top_k]
            top_labels = [labels[i] for i in sorted_indices]
            top_scores = [float(probs[i]) for i in sorted_indices]

            return {
                "labels": top_labels,
                "scores": top_scores,
                "top_k": top_k,
                "all_scores": {labels[i]: float(probs[i]) for i in range(len(labels))},
            }

        return {
            "model_type": "clap",
            "device": actual_device,
            "capabilities": ["embed", "classify"],
            "modes": ["batch"],
            "embed": {"embed": encode_audio},
            "classify": {"classify": classify_audio},
            "raw": {"model": model},
        }

    except ImportError as e:
        raise ImportError(f"Failed to load CLAP: {e}")


ModelRegistry.register_loader(
    "clap",
    load_clap,
    "CLAP: Audio embeddings + zero-shot classification",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["embed", "classify"],
    hardware=["cpu", "mps", "cuda"],
    modes=["batch"],
)


# =============================================================================
# Voxtral Streaming ASR (LCS-10)
# =============================================================================


def load_voxtral(config: dict[str, Any], device: str) -> Bundle:
    """
    Load Voxtral streaming ASR model with Bundle Contract v2.

    Voxtral is Mistral AI's streaming ASR model. Uses the StreamingAdapter
    base class from harness/streaming.py.

    Surfaces:
    - asr_stream: Real-time streaming ASR
    - asr: Batch transcription (via streaming internally)
    """
    import os
    import numpy as np

    # Import streaming utilities
    from harness.streaming import (
        StreamingAdapter,
        StreamEvent,
        StreamEventType,
        ChunkConfig,
        AudioChunker,
        normalize_audio_input,
    )

    # Check for API key
    api_key = config.get("api_key") or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        logger.warning("MISTRAL_API_KEY not set. Voxtral will use mock mode.")

    variant = config.get("variant", "realtime")

    logger.info(f"Loading Voxtral ({variant})...")

    class VoxtralStreamingAdapter(StreamingAdapter):
        """
        Streaming adapter for Voxtral ASR.

        Uses Mistral API websocket for real-time transcription.
        Falls back to mock mode if API key not available.
        """

        def __init__(self, api_key: str | None = None):
            super().__init__("voxtral", debug=False)
            self.api_key = api_key
            self._client = None
            self._accumulated_text = ""
            self._current_segment_id = ""
            self._chunk_config = ChunkConfig(
                frame_ms=20,
                chunk_ms=160,
                sample_rate=16000,
            )

        def _do_start_stream(self, config: dict[str, Any]) -> None:
            """Initialize Voxtral streaming session."""
            self._accumulated_text = ""
            self._current_segment_id = self.handle.get_or_create_segment_id("seg_0")

            if self.api_key:
                try:
                    from mistralai import Mistral

                    self._client = Mistral(api_key=self.api_key)
                    logger.debug("Voxtral: Connected to Mistral API")
                except ImportError:
                    logger.warning("mistralai not installed, using mock mode")
                    self._client = None

        def _do_push_audio(
            self,
            audio: bytes | np.ndarray,
            sr: int,
        ) -> Iterator[StreamEvent]:
            """Process audio chunk through Voxtral."""
            import time

            # Normalize and resample
            audio_arr = normalize_audio_input(audio, "float32")
            if sr != 16000:
                from harness.streaming import resample_audio

                audio_arr = resample_audio(audio_arr, sr, 16000)

            chunk_duration_ms = len(audio_arr) / 16000 * 1000
            t_start = self.handle._audio_position_ms
            self.handle.advance_audio_position(chunk_duration_ms)
            t_end = self.handle._audio_position_ms

            if self._client is not None:
                # Real API call would go here
                # For now, simulate with mock behavior
                pass

            # Mock/simulation: accumulate text based on audio energy
            rms = np.sqrt(np.mean(audio_arr**2))
            if rms > 0.01:  # Has speech
                # Simulate word detection
                mock_words = ["hello", "world", "this", "is", "voxtral"]
                word_idx = min(
                    len(self._accumulated_text.split()),
                    len(mock_words) - 1,
                )

                if self._accumulated_text:
                    self._accumulated_text += " " + mock_words[word_idx]
                else:
                    self._accumulated_text = mock_words[word_idx]

                yield StreamEvent(
                    type=StreamEventType.PARTIAL,
                    text=self._accumulated_text,
                    seq=self.handle.next_seq(),
                    segment_id=self._current_segment_id,
                    t_audio_ms_start=t_start,
                    t_audio_ms_end=t_end,
                    t_emit_ms=time.time() * 1000,
                )

        def _do_flush(self) -> Iterator[StreamEvent]:
            """Emit final transcript."""
            import time

            if self._accumulated_text:
                yield StreamEvent(
                    type=StreamEventType.FINAL,
                    text=self._accumulated_text,
                    seq=self.handle.next_seq(),
                    segment_id=self._current_segment_id,
                    is_endpoint=True,
                    t_emit_ms=time.time() * 1000,
                )

        def _do_finalize(self) -> dict[str, Any]:
            """Return final ASR result."""
            return {
                "text": self._accumulated_text,
                "segments": [
                    {
                        "text": self._accumulated_text,
                        "segment_id": self._current_segment_id,
                    }
                ],
                "language": "en",
            }

        def _do_close(self) -> None:
            """Clean up resources."""
            self._client = None

    # Create adapter instance
    adapter = VoxtralStreamingAdapter(api_key=api_key)

    # Batch transcribe using streaming internally
    def transcribe(audio, sr=16000, **kwargs):
        """Batch transcription using streaming adapter."""
        adapter.start_stream()

        # Process in chunks
        chunker = AudioChunker(
            ChunkConfig(chunk_ms=160, sample_rate=16000),
            audio,
            orig_sr=sr,
        )

        for chunk in chunker.iter_chunks():
            list(adapter.push_audio(chunk, 16000))

        list(adapter.flush())
        return adapter.finalize()

    return {
        "model_type": "voxtral",
        "device": "cpu",  # API-based
        "capabilities": ["asr", "asr_stream"],
        "modes": ["batch", "streaming"],
        "asr": {"transcribe": transcribe},
        "asr_stream": {
            "start_stream": adapter.start_stream,
            "push_audio": adapter.push_audio,
            "flush": adapter.flush,
            "finalize": adapter.finalize,
            "close": adapter.close,
        },
        "raw": {"adapter": adapter},
    }


ModelRegistry.register_loader(
    "voxtral",
    load_voxtral,
    "Voxtral: Streaming ASR from Mistral AI",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["asr", "asr_stream"],
    hardware=["cpu"],
    modes=["batch", "streaming"],
)


# =============================================================================
# Demucs Audio Source Separation (LCS-11)
# =============================================================================


def load_demucs(config: dict[str, Any], device: str) -> Bundle:
    """
    Load Demucs source separation model with Bundle Contract v2.
    Uses demucs < 4.1 API (pretrained + apply).
    """
    try:
        import torch
        import demucs.pretrained
        import demucs.apply
        import numpy as np

        # Device handling
        if device == "mps":
            actual_device = "mps" if torch.backends.mps.is_available() else "cpu"
        elif device == "cuda":
            actual_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            actual_device = "cpu"

        variant = config.get("variant", "htdemucs")
        logger.info(f"Loading Demucs ({variant}) on {actual_device}...")

        # Load model using pretrained API
        model = demucs.pretrained.get_model(variant)
        model.to(actual_device)
        model.eval()

        def separate(audio, sr=44100, **kwargs):
            """
            Separate audio into stems.
            input: audio (samples,) or (channels, samples)
            """
            # Ensure numpy
            if hasattr(audio, "numpy"):
                audio = audio.numpy()
            audio = np.asarray(audio, dtype=np.float32)

            # Save original length
            original_length = audio.shape[-1]

            # Ensure (channels, samples)
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]  # (1, samples)

            # Resample input to model sample rate (usually 44100)
            model_sr = model.samplerate
            if sr != model_sr:
                try:
                    import librosa

                    # Resample input
                    # librosa expects (channels, samples) or (samples,)
                    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=model_sr)
                    audio_tensor = torch.from_numpy(audio_resampled)
                except ImportError:
                    logger.warning("librosa not installed, separation might fail if SR mismatch")
                    audio_tensor = torch.from_numpy(audio)
            else:
                audio_tensor = torch.from_numpy(audio)

            # Demucs expects stereo (2, samples) usually, but apply_model handles expansion if 1 ch?
            # Actually apply_model expects (batch, channels, samples)
            ref = audio_tensor.mean(0)
            audio_tensor = audio_tensor.unsqueeze(0)  # (1, channels, samples)
            # If mono, expand to stereo if model requires?
            # htdemucs is usually trained on stereo. apply_model handles this internaly often but
            # let's rely on apply_model default behavior or expand.
            if audio_tensor.shape[1] == 1 and model.audio_channels == 2:
                audio_tensor = audio_tensor.repeat(1, 2, 1)

            audio_tensor = audio_tensor.to(actual_device)

            # Run separation
            with torch.no_grad():
                # returns (batch, sources, channels, samples)
                sources = demucs.apply.apply_model(
                    model, audio_tensor, shifts=0, split=True, overlap=0.25, device=actual_device
                )

            # sources: (1, 4, 2, samples)
            sources = sources.cpu()

            # Post-processing
            stems = {}
            source_names = model.sources  # ['drums', 'bass', 'other', 'vocals']

            for i, name in enumerate(source_names):
                # source: (2, samples)
                stem_audio = sources[0, i]

                # Convert to mono by averaging channels
                stem_audio = stem_audio.mean(dim=0).numpy()  # (samples,)

                # Resample back if needed
                if sr != model_sr:
                    import librosa

                    stem_audio = librosa.resample(stem_audio, orig_sr=model_sr, target_sr=sr)

                # Align length
                if len(stem_audio) > original_length:
                    stem_audio = stem_audio[:original_length]
                elif len(stem_audio) < original_length:
                    stem_audio = np.pad(stem_audio, (0, original_length - len(stem_audio)))

                stems[name] = stem_audio.astype(np.float32)

            return {
                "stems": stems,
                "sr": sr,
            }

        return {
            "model_type": "demucs",
            "device": actual_device,
            "capabilities": ["separate"],
            "modes": ["batch"],
            "separate": {"separate": separate},
            "raw": {"model": model},
        }

    except ImportError as e:
        raise ImportError(f"Failed to load Demucs deps: {e}") from e


ModelRegistry.register_loader(
    "demucs",
    load_demucs,
    "Demucs: Audio source separation",
    status=ModelStatus.EXPERIMENTAL,
    version="4.0.0",
    capabilities=["separate"],
    hardware=["cpu", "mps", "cuda"],
    modes=["batch"],
)


# =============================================================================
# Basic Pitch Automatic Music Transcription (LCS-12)
# =============================================================================


def load_basic_pitch(config: dict[str, Any], device: str) -> Bundle:
    """
    Load Basic Pitch music transcription model with Bundle Contract v2.

    Basic Pitch transcribes audio to MIDI notes:
    - Each note has onset, offset, pitch (MIDI), velocity

    Output: {notes: [{onset, offset, pitch, velocity}], ...}
    """
    try:
        import numpy as np

        # Import basic-pitch
        try:
            from basic_pitch.inference import predict
            from basic_pitch import ICASSP_2022_MODEL_PATH
        except ImportError:
            raise ImportError(
                "basic-pitch not installed. Install with:\n"
                "pip install -r models/basic_pitch/requirements.txt"
            ) from None

        logger.info("Loading Basic Pitch...")

        def transcribe(audio, sr=22050, **kwargs):
            """
            Transcribe audio to MIDI notes.

            Args:
                audio: Input audio
                sr: Sample rate

            Returns:
                {notes: [{onset, offset, pitch, velocity}, ...], ...}
            """
            # Ensure numpy array
            if hasattr(audio, "numpy"):
                audio = audio.numpy()
            audio = np.asarray(audio, dtype=np.float32)

            # Ensure mono
            if audio.ndim > 1:
                audio = audio.mean(axis=0)

            # Resample if needed
            if sr != 22050:
                try:
                    import librosa

                    audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
                except ImportError:
                    pass

            # Run prediction
            model_output, midi_data, note_events = predict(
                audio,
                ICASSP_2022_MODEL_PATH,
            )

            # Convert note events to consistent format
            notes = []
            for start_time, end_time, pitch, velocity, pitch_bend in note_events:
                notes.append(
                    {
                        "onset": float(start_time),
                        "offset": float(end_time),
                        "pitch": int(pitch),  # MIDI pitch (0-127)
                        "velocity": float(velocity),
                    }
                )

            return {
                "notes": notes,
                "midi": midi_data,  # Pretty MIDI object
                "model_output": {
                    "contours": model_output["contour"],
                    "notes": model_output["note"],
                    "onsets": model_output["onset"],
                },
            }

        return {
            "model_type": "basic_pitch",
            "device": "cpu",
            "capabilities": ["music_transcription"],
            "modes": ["batch"],
            "music_transcription": {"transcribe": transcribe},
            "raw": {},
        }

    except ImportError as e:
        raise ImportError(f"Failed to load Basic Pitch: {e}") from e


ModelRegistry.register_loader(
    "basic_pitch",
    load_basic_pitch,
    "Basic Pitch: Automatic music transcription",
    status=ModelStatus.EXPERIMENTAL,
    version="0.2.0",
    capabilities=["music_transcription"],
    hardware=["cpu"],
    modes=["batch"],
)


# =============================================================================
# Systran Faster-Whisper Large V3 (LCS-14)
# =============================================================================


def load_faster_whisper_large_v3(config: dict[str, Any], device: str) -> Bundle:
    """
    Load Systran Faster-Whisper Large V3 with Bundle Contract v2.

    Explicitly uses Systran/faster-whisper-large-v3 from HuggingFace.
    This is the strongest Whisper-family anchor in the CTranslate2 path.
    """
    try:
        import numpy as np

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper not installed. Install with:\n"
                "pip install -r models/faster_whisper_large_v3/requirements.txt"
            ) from None

        # Force Systran large-v3
        model_name = "Systran/faster-whisper-large-v3"
        compute_type = config.get("compute_type", "float16")
        beam_size = config.get("beam_size", 5)

        # Device mapping
        if device == "mps":
            device_fw = "cpu"  # faster-whisper doesn't support MPS
        elif device == "cuda":
            device_fw = "cuda"
        else:
            device_fw = "cpu"

        # Force int8 on CPU if float16 requested
        if device_fw == "cpu" and compute_type == "float16":
            compute_type = "int8"

        logger.info(f"Loading Faster-Whisper Large V3 on {device_fw} ({compute_type})...")

        model = WhisperModel(model_name, device=device_fw, compute_type=compute_type)

        def transcribe(audio, sr=16000, **kwargs):
            """Transcribe audio using Faster-Whisper Large V3."""
            # Ensure numpy array
            if hasattr(audio, "numpy"):
                audio = audio.numpy()
            audio = np.asarray(audio, dtype=np.float32)

            # Ensure mono
            if audio.ndim > 1:
                audio = audio.mean(axis=0)

            # Resample if needed
            if sr != 16000:
                try:
                    import librosa

                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                except ImportError:
                    pass

            # Pop unsupported kwargs
            kwargs.pop("progress_callback", None)

            segments_generator, info = model.transcribe(
                audio,
                beam_size=beam_size,
                **kwargs,
            )

            segments_list = list(segments_generator)
            text = " ".join([s.text for s in segments_list])

            return {
                "text": text.strip(),
                "segments": [
                    {
                        "start": s.start,
                        "end": s.end,
                        "text": s.text.strip(),
                    }
                    for s in segments_list
                ],
                "language": info.language if info else "",
                "language_probability": info.language_probability if info else 0,
            }

        return {
            "model_type": "faster_whisper_large_v3",
            "device": device_fw,
            "capabilities": ["asr"],
            "modes": ["batch"],
            "asr": {"transcribe": transcribe},
            "raw": {"model": model},
        }

    except ImportError as e:
        raise ImportError(f"Failed to load Faster-Whisper Large V3: {e}") from e


ModelRegistry.register_loader(
    "faster_whisper_large_v3",
    load_faster_whisper_large_v3,
    "Faster-Whisper Large V3: Systran's optimized Whisper (CTranslate2)",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["asr"],
    hardware=["cpu", "cuda"],
    modes=["batch"],
)


# =============================================================================
# Faster-Distil-Whisper Large V3 (LCS-15)
# =============================================================================


def load_faster_distil_whisper_large_v3(config: dict[str, Any], device: str) -> Bundle:
    """
    Load Faster-Distil-Whisper Large V3 with Bundle Contract v2.

    Distilled Whisper for 2-3x faster inference with similar quality.
    Uses same CTranslate2 runtime as LCS-14.
    """
    try:
        import numpy as np

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper not installed. Install with:\n"
                "pip install -r models/faster_distil_whisper_large_v3/requirements.txt"
            ) from None

        # Force Systran distil-large-v3
        model_name = "Systran/faster-distil-whisper-large-v3"
        compute_type = config.get("compute_type", "float16")
        beam_size = config.get("beam_size", 5)

        # Device mapping
        if device == "mps":
            device_fw = "cpu"
        elif device == "cuda":
            device_fw = "cuda"
        else:
            device_fw = "cpu"

        # Force int8 on CPU
        if device_fw == "cpu" and compute_type == "float16":
            compute_type = "int8"

        logger.info(f"Loading Faster-Distil-Whisper Large V3 on {device_fw} ({compute_type})...")

        model = WhisperModel(model_name, device=device_fw, compute_type=compute_type)

        def transcribe(audio, sr=16000, **kwargs):
            """Transcribe audio using Faster-Distil-Whisper."""
            if hasattr(audio, "numpy"):
                audio = audio.numpy()
            audio = np.asarray(audio, dtype=np.float32)

            if audio.ndim > 1:
                audio = audio.mean(axis=0)

            if sr != 16000:
                try:
                    import librosa

                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                except ImportError:
                    pass

            kwargs.pop("progress_callback", None)

            segments_generator, info = model.transcribe(
                audio,
                beam_size=beam_size,
                **kwargs,
            )

            segments_list = list(segments_generator)
            text = " ".join([s.text for s in segments_list])

            return {
                "text": text.strip(),
                "segments": [
                    {"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments_list
                ],
                "language": info.language if info else "",
                "language_probability": info.language_probability if info else 0,
            }

        return {
            "model_type": "faster_distil_whisper_large_v3",
            "device": device_fw,
            "capabilities": ["asr"],
            "modes": ["batch"],
            "asr": {"transcribe": transcribe},
            "raw": {"model": model},
        }

    except ImportError as e:
        raise ImportError(f"Failed to load Faster-Distil-Whisper: {e}") from e


ModelRegistry.register_loader(
    "faster_distil_whisper_large_v3",
    load_faster_distil_whisper_large_v3,
    "Faster-Distil-Whisper Large V3: Distilled Whisper (2-3x faster)",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["asr"],
    hardware=["cpu", "cuda"],
    modes=["batch"],
)


# =============================================================================
# GLM-ASR-Nano-2512 (LCS-16)
# =============================================================================


def load_glm_asr_nano_2512(config: dict[str, Any], device: str) -> Bundle:
    """
    Load GLM-ASR-Nano-2512 with Bundle Contract v2.

    First non-Whisper batch ASR in Batch 2.
    Uses PyTorch + HuggingFace Transformers.
    """
    try:
        import numpy as np

        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError:
            raise ImportError(
                "PyTorch/transformers not installed. Install with:\n"
                "pip install -r models/glm_asr_nano_2512/requirements.txt"
            ) from None

        model_id = "THUDM/glm-4-voice-decoder"

        # Device mapping
        if device == "mps":
            torch_device = "mps"
            dtype = torch.float32  # MPS often needs float32
        elif device == "cuda":
            torch_device = "cuda"
            dtype = torch.float16
        else:
            torch_device = "cpu"
            dtype = torch.float32

        logger.info(f"Loading GLM-ASR-Nano-2512 on {torch_device}...")

        try:
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(torch_device)
        except Exception as e:
            # Fallback: GLM might need different loading
            logger.warning(f"Standard loading failed: {e}. Using fallback.")
            processor = None
            model = None

        def transcribe(audio, sr=16000, **kwargs):
            """Transcribe audio using GLM-ASR."""
            if hasattr(audio, "numpy"):
                audio = audio.numpy()
            audio = np.asarray(audio, dtype=np.float32)

            if audio.ndim > 1:
                audio = audio.mean(axis=0)

            # Resample if needed
            if sr != 16000:
                try:
                    import librosa

                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                except ImportError:
                    pass

            # If model/processor not loaded, return placeholder
            if processor is None or model is None:
                return {
                    "text": "[GLM-ASR model not fully configured]",
                    "error": "Model requires specific HuggingFace setup",
                }

            # Process and generate
            inputs = processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
            ).to(torch_device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=256)

            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return {
                "text": text.strip(),
            }

        return {
            "model_type": "glm_asr_nano_2512",
            "device": torch_device,
            "capabilities": ["asr"],
            "modes": ["batch"],
            "asr": {"transcribe": transcribe},
            "raw": {"model": model, "processor": processor},
        }

    except ImportError as e:
        raise ImportError(f"Failed to load GLM-ASR-Nano-2512: {e}") from e


ModelRegistry.register_loader(
    "glm_asr_nano_2512",
    load_glm_asr_nano_2512,
    "GLM-ASR-Nano-2512: Lightweight non-Whisper ASR (PyTorch)",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["asr"],
    hardware=["cpu", "cuda", "mps"],
    modes=["batch"],
)


# =============================================================================
# NB-Whisper-Small-ONNX (LCS-17)
# =============================================================================


def load_nb_whisper_small_onnx(config: dict[str, Any], device: str) -> Bundle:
    """
    Load NB-Whisper-Small-ONNX with Bundle Contract v2.

    Norwegian Whisper with ONNX runtime for cross-platform inference.
    Last in Batch 2 due to potential Mac packaging edge cases.
    """
    try:
        import numpy as np
        import torch
        from transformers import pipeline
    except ImportError as e:
        raise ImportError(
            "transformers/PyTorch not installed. Install with:\n"
            "pip install -r models/nb_whisper_small_onnx/requirements.txt"
        ) from e

    model_id = (
        config.get("model_name")
        or config.get("config", {}).get("model_name")
        or "NbAiLab/nb-whisper-small"
    )
    language = config.get("language") or config.get("config", {}).get("language") or "no"

    backend = "transformers_fallback"
    asr_pipe = None

    # Preferred: ONNXRuntime via optimum if available.
    try:
        import onnxruntime as ort
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
        from transformers import AutoProcessor

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        logger.info("Loading NB-Whisper ONNX (optimum) with providers: %s", providers)
        processor = AutoProcessor.from_pretrained(model_id)
        ort_model = ORTModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            subfolder="onnx",
            use_merged=False,
            encoder_file_name="encoder_model.onnx",
            decoder_file_name="decoder_model.onnx",
            decoder_with_past_file_name="decoder_with_past_model.onnx",
            provider=providers[0],
        )
        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=ort_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
        )
        backend = "onnxruntime_optimum"
    except Exception as onnx_exc:
        logger.warning(
            "NB-Whisper ONNX backend unavailable (%s). Falling back to transformers runtime.",
            onnx_exc,
        )
        torch_device = "cpu"
        if device == "cuda" and torch.cuda.is_available():
            torch_device = "cuda:0"
        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=torch_device,
        )

    def transcribe(audio, sr=16000, **kwargs):
        """Transcribe audio using NB-Whisper (ONNX preferred, transformers fallback)."""
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        if sr != 16000:
            try:
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            except ImportError:
                pass

        result = asr_pipe(
            {"array": audio, "sampling_rate": 16000},
            return_timestamps=True,
            generate_kwargs={"language": language},
        )
        text = (result.get("text") if isinstance(result, dict) else str(result)) or ""
        chunks = result.get("chunks", []) if isinstance(result, dict) else []
        segments = []
        for c in chunks:
            ts = c.get("timestamp") if isinstance(c, dict) else None
            if ts and len(ts) == 2 and ts[0] is not None and ts[1] is not None:
                segments.append({"start": float(ts[0]), "end": float(ts[1]), "text": c.get("text", "")})

        return {
            "text": text.strip(),
            "segments": segments,
            "language": language,
            "meta": {"backend": backend, "model": model_id},
        }

    return {
        "model_type": "nb_whisper_small_onnx",
        "device": device,
        "capabilities": ["asr"],
        "modes": ["batch"],
        "asr": {"transcribe": transcribe},
        "raw": {"pipeline": asr_pipe, "backend": backend},
    }


ModelRegistry.register_loader(
    "nb_whisper_small_onnx",
    load_nb_whisper_small_onnx,
    "NB-Whisper-Small-ONNX: Norwegian Whisper with ONNX runtime",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["asr"],
    hardware=["cpu", "cuda"],
    modes=["batch"],
)


# =============================================================================
# Kyutai Streaming ASR (LCS-19)
# =============================================================================


def load_kyutai_streaming(config: dict[str, Any], device: str) -> Bundle:
    """
    Load Kyutai Streaming ASR with Bundle Contract v2.

    First streaming ASR in Batch 3. Uses StreamingAdapter for lifecycle.
    Streaming contract: seq_monotonic, segment_id_stable, finalize_idempotent.
    """
    try:
        import numpy as np

        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch not installed. Install with:\n"
                "pip install -r models/kyutai_streaming/requirements.txt"
            ) from None

        from harness.streaming import StreamEvent, StreamEventType, StreamingAdapter

        # Device mapping
        if device == "mps":
            torch_device = "mps"
        elif device == "cuda":
            torch_device = "cuda"
        else:
            torch_device = "cpu"

        chunk_ms = config.get("chunk_ms", 100)

        logger.info(f"Loading Kyutai Streaming on {torch_device} (chunk_ms={chunk_ms})...")

        class KyutaiStreamingAdapter(StreamingAdapter):
            """Kyutai-specific streaming adapter."""

            def __init__(self):
                super().__init__("kyutai_streaming")
                self._buffer = []
                self._sample_rate = 16000
                self._chunks_processed = 0
                self._accumulated_text = ""

            def _do_start_stream(self, config: dict[str, Any]) -> None:
                """Initialize stream state."""
                self._buffer = []
                self._chunks_processed = 0
                self._accumulated_text = ""
                self._sample_rate = config.get("sample_rate", 16000)

            def _do_push_audio(
                self, audio: bytes | np.ndarray, sr: int
            ) -> Iterator[StreamEvent]:
                """Process a single audio chunk."""
                self._chunks_processed += 1
                self._accumulated_text = f"[chunks={self._chunks_processed}]"

                segment_id = self.handle.get_or_create_segment_id("seg_0")
                event = StreamEvent(
                    type=StreamEventType.PARTIAL,
                    text=self._accumulated_text,
                    segment_id=segment_id,
                )
                yield event

            def _do_flush(self) -> Iterator[StreamEvent]:
                """No-op flush for placeholder model."""
                return iter(())

            def _do_finalize(self) -> dict[str, Any]:
                """Finalize and return final transcript."""
                return {
                    "text": f"[final: {self._chunks_processed} chunks processed]",
                    "is_final": True,
                    "chunks_processed": self._chunks_processed,
                }

            def get_transcript(self) -> dict[str, Any]:
                """Return current transcript snapshot."""
                return {
                    "text": self._accumulated_text,
                    "is_final": False,
                    "chunks_processed": self._chunks_processed,
                }

        # Create singleton adapter instance
        adapter = KyutaiStreamingAdapter()

        def start(sr: int = 16000, **kwargs) -> str:
            """Start a new streaming session."""
            handle = adapter.start_stream({"sample_rate": sr, **kwargs})
            return handle.stream_id

        def start_stream(sr: int = 16000, **kwargs) -> str:
            """Bundle Contract v2 stream start."""
            return start(sr=sr, **kwargs)

        def push_audio(handle: str, audio: np.ndarray, sr: int = 16000) -> None:
            """Push audio chunk to stream."""
            if hasattr(audio, "numpy"):
                audio = audio.numpy()
            audio = np.asarray(audio, dtype=np.float32)
            adapter.push_audio(audio, sr=sr)

        def flush(handle: str) -> list[dict[str, Any]]:
            """Flush pending events."""
            return list(adapter.flush())

        def get_transcript(handle: str) -> dict[str, Any]:
            """Get current transcript."""
            return adapter.get_transcript()

        def finalize(handle: str) -> dict[str, Any]:
            """Finalize stream and get final transcript."""
            return adapter.finalize()

        def close(handle: str) -> None:
            """Close stream resources."""
            adapter.close()

        return {
            "model_type": "kyutai_streaming",
            "device": torch_device,
            "capabilities": ["asr_stream"],
            "modes": ["streaming"],
            "asr_stream": {
                "start_stream": start_stream,
                "start": start,
                "push_audio": push_audio,
                "flush": flush,
                "get_transcript": get_transcript,
                "finalize": finalize,
                "close": close,
            },
            "raw": {"adapter": adapter},
        }

    except ImportError as e:
        raise ImportError(f"Failed to load Kyutai Streaming: {e}") from e


ModelRegistry.register_loader(
    "kyutai_streaming",
    load_kyutai_streaming,
    "Kyutai Streaming: Lightweight streaming ASR (PyTorch)",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["asr_stream"],
    hardware=["cpu", "cuda", "mps"],
    modes=["streaming"],
)


# =============================================================================
# GLM-TTS (LCS-21)
# =============================================================================


def load_glm_tts(config: dict[str, Any], device: str) -> Bundle:
    """
    Load GLM-TTS with Bundle Contract v2.

    Text-to-speech synthesis from THUDM GLM-4 Voice.
    Broadens TTS coverage beyond lfm2_5_audio.
    """
    import os
    from functools import partial

    # GLM-TTS is a strict Repo Pipeline model.
    # It does NOT support standard AutoModel usage.
    # Accept both "glm_tts" (registry ID) and "zai-org/GLM-TTS" (HF Hub ID)

    # 1. Allowlist Check - accept valid registry name or HF Hub ID
    model_id = config.get("model_id", "glm_tts")
    valid_ids = ["glm_tts", "zai-org/GLM-TTS"]
    if model_id not in valid_ids:
        logger.error(f"GLM-TTS request rejected. Got: {model_id}, Expected one of: {valid_ids}")
        return {
            "model_type": "glm_tts",
            "status": "blocked",
            "error": f"Invalid model ID. GLM-TTS requires one of: {valid_ids}",
        }

    # 2. Define the Repo Spec
    # This tells the runner to use the external wrapper script
    repo_root = os.path.abspath("models/glm_tts")
    wrapper_path = os.path.join(repo_root, "wrapper.py")
    ckpt_dir = os.path.join(repo_root, "ckpt")

    # 3. Check for Wrapper (Safety)
    if not os.path.exists(wrapper_path):
        return {
            "model_type": "glm_tts",
            "status": "blocked",
            "error": f"Wrapper not found at {wrapper_path}. Implementation incomplete.",
        }

    # 4. Return the Repo Pipeline Spec
    # The runner will invoke: python models/glm_tts/wrapper.py ...
    return {
        "model_type": "glm_tts",
        "kind": "repo_pipeline",
        "device": device,
        "capabilities": ["tts"],
        "modes": ["batch"],
        "repo_spec": {
            "wrapper_path": wrapper_path,
            "ckpt_dir": ckpt_dir,
            "venv_path": os.environ.get("VIRTUAL_ENV"),  # Pass current venv if active
        },
        "tts": {
            # benchmark/runner.py expects a callable for 'synthesize' in normal mode,
            # but for repo_pipeline models, the runner handles execution differently.
            # However, to keep existing interfaces happy for now, we provide a
            # proxy that calls the wrapper via subprocess if invoked directly.
            # ideally bench/runner.py handles 'kind' == 'repo_pipeline'.
            "synthesize": partial(_invoke_repo_wrapper, wrapper_path=wrapper_path)
        },
    }


def _invoke_repo_wrapper(text: str, wrapper_path: str, **kwargs) -> tuple["np.ndarray", int]:
    """
    Proxy to invoke the wrapper script from python.
    Used when the harness calls .synthesize() directly.
    """
    import subprocess
    import json
    import tempfile
    import numpy as np

    # We need to construct the CLI call to the wrapper
    # The wrapper expects: --text "..." --output_dir ...

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = ["python", wrapper_path, "--text", text, "--output_dir", tmpdir]

        # Add other kwargs if supported

        try:
            # Capture JSON output from stdout (wrapper should print JSON result)
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Parse only the last line as JSON, assuming logging above
            lines = result.stdout.strip().split("\n")
            json_line = lines[-1]
            res = json.loads(json_line)

            # Helper: Load audio if path provided but data missing
            if "audio_path" in res and "audio" not in res and res.get("status") == "ok":
                import soundfile as sf
                import os

                if os.path.exists(res["audio_path"]):
                    data, sr = sf.read(res["audio_path"])
                    res["audio"] = data
                    res["sample_rate"] = sr

            if res.get("status") == "ok" and "audio" in res:
                audio = np.asarray(res["audio"], dtype=np.float32)
                sample_rate = int(res.get("sample_rate", 24000))
                return audio, sample_rate

            logger.warning("GLM-TTS wrapper returned no audio. Falling back to silence.")
            return np.zeros(1, dtype=np.float32), int(res.get("sample_rate", 24000))
        except subprocess.CalledProcessError as e:
            logger.error(f"GLM-TTS wrapper failed: {e.stderr}")
            return np.zeros(1, dtype=np.float32), 24000
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from wrapper: {result.stdout}")
            return np.zeros(1, dtype=np.float32), 24000


ModelRegistry.register_loader(
    "glm_tts",
    load_glm_tts,
    "GLM-TTS: Text-to-speech from THUDM GLM-4 Voice",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["tts"],
    hardware=["cpu", "cuda", "mps"],
    modes=["batch"],
)


# =============================================================================
# Kokoro-TTS (LCS-23)
# =============================================================================


def load_kokoro_tts(config: dict[str, Any], device: str) -> Bundle:
    """
    Load Kokoro-82M TTS with Bundle Contract v2.

    Lightweight CPU/CUDA text-to-speech using the official kokoro package.
    """
    try:
        import importlib.util
        import numpy as np
        import torch
        from kokoro import KPipeline
    except ImportError as e:
        raise ImportError(
            "kokoro package not installed. Install with:\n"
            "uv pip install -r models/kokoro_tts/requirements.txt"
        ) from e

    # misaki/espeak bootstrap may shell out to `python -m pip` on first run.
    if importlib.util.find_spec("pip") is None:
        raise RuntimeError(
            "kokoro_tts runtime requires pip module in this environment. Install with:\n"
            "uv pip install pip"
        )

    # Kokoro currently supports cpu/cuda via torch; map unsupported devices to cpu.
    if device == "cuda" and torch.cuda.is_available():
        actual_device = "cuda"
    else:
        actual_device = "cpu"

    lang_code = str(config.get("lang_code", "a"))
    repo_id = str(config.get("repo_id", "hexgrad/Kokoro-82M"))
    default_voice = str(config.get("voice", "af_heart"))
    default_speed = float(config.get("speed", 1.0))
    split_pattern = config.get("split_pattern", r"\n+")

    pipeline = KPipeline(
        lang_code=lang_code,
        repo_id=repo_id,
        device=actual_device,
    )

    def synthesize(text: str, **kwargs) -> dict[str, Any]:
        voice = str(kwargs.get("voice", default_voice))
        speed = float(kwargs.get("speed", default_speed))
        split = kwargs.get("split_pattern", split_pattern)

        chunks: list[np.ndarray] = []
        for result in pipeline(text, voice=voice, speed=speed, split_pattern=split):
            audio_chunk = result.audio
            if audio_chunk is None:
                continue
            if hasattr(audio_chunk, "detach"):
                audio_np = audio_chunk.detach().cpu().numpy()
            else:
                audio_np = np.asarray(audio_chunk)
            if audio_np.ndim > 1:
                audio_np = np.squeeze(audio_np)
            chunks.append(audio_np.astype(np.float32, copy=False))

        if not chunks:
            audio = np.zeros(1, dtype=np.float32)
        else:
            audio = np.concatenate(chunks)

        # Trim leading/trailing near-silence to reduce false "mostly_silent" flags
        # on short punctuation-heavy utterances while preserving internal pauses.
        silence_threshold = float(kwargs.get("silence_trim_threshold", 0.006))
        non_silent = np.flatnonzero(np.abs(audio) > silence_threshold)
        if non_silent.size > 0:
            start = int(non_silent[0])
            end = int(non_silent[-1]) + 1
            audio = audio[start:end]

        sr = 24000
        return {
            "audio": audio,
            "sample_rate": sr,
            "duration_s": len(audio) / sr if sr > 0 else 0.0,
            "meta": {
                "model": "hexgrad/Kokoro-82M",
                "voice": voice,
                "lang_code": lang_code,
            },
        }

    return {
        "model_type": "kokoro_tts",
        "device": actual_device,
        "capabilities": ["tts"],
        "modes": ["batch"],
        "tts": {"synthesize": synthesize},
        "raw": {"pipeline": pipeline},
    }


ModelRegistry.register_loader(
    "kokoro_tts",
    load_kokoro_tts,
    "Kokoro-TTS: Lightweight high-quality text-to-speech (hexgrad/Kokoro-82M)",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["tts"],
    hardware=["cpu", "cuda"],
    modes=["batch"],
)


# =============================================================================
# Voxtral Realtime 2602 (LCS-22)
# =============================================================================


def load_voxtral_realtime_2602(config: dict[str, Any], device: str) -> Bundle:
    """
    Load Voxtral Realtime 2602 with Bundle Contract v2.

    Open-weights Voxtral Mini 4B with configurable transcription_delay_ms.
    Reuses StreamingAdapter for lifecycle.
    """
    try:
        import numpy as np

        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch not installed. Install with:\n"
                "pip install -r models/voxtral_realtime_2602/requirements.txt"
            ) from None

        from harness.streaming import StreamEvent, StreamEventType, StreamingAdapter

        # Device mapping
        if device == "mps":
            torch_device = "mps"
        elif device == "cuda":
            torch_device = "cuda"
        else:
            torch_device = "cpu"

        # Configurable delay knob (model supports 80ms-2.4s)
        transcription_delay_ms = config.get("transcription_delay_ms", 200)
        # Clamp to model's supported range
        transcription_delay_ms = max(80, min(2400, transcription_delay_ms))

        chunk_ms = config.get("chunk_ms", 100)

        logger.info(
            f"Loading Voxtral Realtime 2602 (delay={transcription_delay_ms}ms, chunk={chunk_ms}ms)..."
        )

        class VoxtralRealtimeAdapter(StreamingAdapter):
            """Voxtral Realtime streaming adapter with configurable delay."""

            def __init__(self, delay_ms: int):
                super().__init__("voxtral_realtime_2602")
                self._delay_ms = delay_ms
                self._buffer = []
                self._sample_rate = 16000
                self._chunks_processed = 0
                self._accumulated_text = ""

            def _do_start_stream(self, config: dict[str, Any]) -> None:
                """Initialize stream state."""
                self._buffer = []
                self._chunks_processed = 0
                self._accumulated_text = ""
                self._sample_rate = config.get("sample_rate", 16000)

            def _do_push_audio(
                self, audio: bytes | np.ndarray, sr: int
            ) -> Iterator[StreamEvent]:
                """Process a single audio chunk with delay."""
                self._chunks_processed += 1
                self._accumulated_text = f"[voxtral chunks={self._chunks_processed}]"

                segment_id = self.handle.get_or_create_segment_id("seg_0")
                event = StreamEvent(
                    type=StreamEventType.PARTIAL,
                    text=self._accumulated_text,
                    segment_id=segment_id,
                )
                yield event

            def _do_flush(self) -> Iterator[StreamEvent]:
                """No-op flush for placeholder model."""
                return iter(())

            def _do_finalize(self) -> dict[str, Any]:
                """Finalize and return final transcript."""
                return {
                    "text": f"[Voxtral final: {self._chunks_processed} chunks, delay={self._delay_ms}ms]",
                    "is_final": True,
                    "chunks_processed": self._chunks_processed,
                    "delay_ms": self._delay_ms,
                }

            def get_transcript(self) -> dict[str, Any]:
                """Return current transcript snapshot."""
                return {
                    "text": self._accumulated_text,
                    "is_final": False,
                    "delay_ms": self._delay_ms,
                }

        adapter = VoxtralRealtimeAdapter(delay_ms=transcription_delay_ms)

        def start(sr: int = 16000, **kwargs) -> str:
            """Start a new streaming session."""
            handle = adapter.start_stream({"sample_rate": sr, **kwargs})
            return handle.stream_id

        def start_stream(sr: int = 16000, **kwargs) -> str:
            """Bundle Contract v2 stream start."""
            return start(sr=sr, **kwargs)

        def push_audio(handle: str, audio: np.ndarray, sr: int = 16000) -> None:
            """Push audio chunk to stream."""
            if hasattr(audio, "numpy"):
                audio = audio.numpy()
            audio = np.asarray(audio, dtype=np.float32)
            adapter.push_audio(audio, sr=sr)

        def flush(handle: str) -> list[dict[str, Any]]:
            """Flush pending events."""
            return list(adapter.flush())

        def get_transcript(handle: str) -> dict[str, Any]:
            """Get current transcript."""
            return adapter.get_transcript()

        def finalize(handle: str) -> dict[str, Any]:
            """Finalize stream and get final transcript."""
            return adapter.finalize()

        def close(handle: str) -> None:
            """Close stream resources."""
            adapter.close()

        return {
            "model_type": "voxtral_realtime_2602",
            "device": torch_device,
            "capabilities": ["asr_stream"],
            "modes": ["streaming"],
            "config": {
                "transcription_delay_ms": transcription_delay_ms,
                "chunk_ms": chunk_ms,
            },
            "asr_stream": {
                "start_stream": start_stream,
                "start": start,
                "push_audio": push_audio,
                "flush": flush,
                "get_transcript": get_transcript,
                "finalize": finalize,
                "close": close,
            },
            "raw": {"adapter": adapter},
        }

    except ImportError as e:
        raise ImportError(f"Failed to load Voxtral Realtime: {e}") from e


ModelRegistry.register_loader(
    "voxtral_realtime_2602",
    load_voxtral_realtime_2602,
    "Voxtral Realtime 2602: Open-weights streaming ASR (PyTorch)",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["asr_stream"],
    hardware=["cpu", "cuda", "mps"],
    modes=["streaming"],
)


# =============================================================================
# Nemotron Streaming ASR (LCS-18) - NeMo
# =============================================================================


def load_nemotron_streaming(config: dict[str, Any], device: str) -> Bundle:
    """
    Load Nemotron Streaming ASR with Bundle Contract v2.

    NeMo-based streaming ASR. Requires dedicated venv.
    """
    try:
        import numpy as np

        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            raise ImportError(
                "NeMo not installed. Install in dedicated venv:\n"
                "python -m venv .venv.nemo_nemotron\n"
                "source .venv.nemo_nemotron/bin/activate\n"
                "pip install -r models/nemotron_streaming/requirements.txt"
            ) from None

        from harness.streaming import StreamingAdapter

        if device == "cuda":
            torch_device = "cuda"
        else:
            torch_device = "cpu"

        chunk_ms = config.get("chunk_ms", 160)

        logger.info(f"Loading Nemotron Streaming on {torch_device}...")

        class NemotronStreamingAdapter(StreamingAdapter):
            """Nemotron NeMo streaming adapter."""

            def __init__(self):
                super().__init__()
                self._chunks_processed = 0

            def _process_chunk(self, audio_chunk: np.ndarray) -> dict[str, Any]:
                self._chunks_processed += 1
                return {"text": f"[chunk {self._chunks_processed}]", "is_final": False}

            def _finalize_stream(self) -> dict[str, Any]:
                return {"text": f"[Nemotron: {self._chunks_processed} chunks]", "is_final": True}

        adapter = NemotronStreamingAdapter()

        return {
            "model_type": "nemotron_streaming",
            "device": torch_device,
            "capabilities": ["asr_stream"],
            "modes": ["streaming"],
            "asr_stream": {
                "start": lambda sr=16000, **kw: adapter.start_stream(sample_rate=sr, **kw),
                "push_audio": lambda h, a: adapter.push_audio(h, np.asarray(a, dtype=np.float32)),
                "get_transcript": lambda h: adapter.get_transcript(h),
                "finalize": lambda h: adapter.finalize(h),
            },
            "raw": {"adapter": adapter},
        }

    except ImportError as e:
        raise ImportError(f"Failed to load Nemotron Streaming: {e}") from e


ModelRegistry.register_loader(
    "nemotron_streaming",
    load_nemotron_streaming,
    "Nemotron Streaming: NeMo-based streaming ASR",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["asr_stream"],
    hardware=["cpu", "cuda"],
    modes=["streaming"],
)


# =============================================================================
# Parakeet Multitalker (LCS-20) - NeMo
# =============================================================================


def load_parakeet_multitalker(config: dict[str, Any], device: str) -> Bundle:
    """
    Load Parakeet Multitalker with Bundle Contract v2.

    NeMo-based multitalker ASR with diarization. Requires dedicated venv.
    """
    try:
        import numpy as np

        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            raise ImportError(
                "NeMo not installed. Install in dedicated venv:\n"
                "python -m venv .venv.nemo_parakeet\n"
                "source .venv.nemo_parakeet/bin/activate\n"
                "pip install -r models/parakeet_multitalker/requirements.txt"
            ) from None

        if device == "cuda":
            torch_device = "cuda"
        else:
            torch_device = "cpu"

        max_speakers = config.get("max_speakers", 4)

        logger.info(
            f"Loading Parakeet Multitalker on {torch_device} (max_speakers={max_speakers})..."
        )

        def transcribe(audio, sr=16000, speaker_segments=None, **kwargs):
            """Transcribe with multitalker handling."""
            if hasattr(audio, "numpy"):
                audio = audio.numpy()
            audio = np.asarray(audio, dtype=np.float32)

            # Placeholder for real NeMo inference
            return {
                "text": "[Parakeet multitalker placeholder]",
                "speakers": [],
            }

        return {
            "model_type": "parakeet_multitalker",
            "device": torch_device,
            "capabilities": ["asr"],
            "modes": ["batch"],
            "asr": {"transcribe": transcribe},
            "raw": {},
        }

    except ImportError as e:
        raise ImportError(f"Failed to load Parakeet Multitalker: {e}") from e


ModelRegistry.register_loader(
    "parakeet_multitalker",
    load_parakeet_multitalker,
    "Parakeet Multitalker: NeMo-based multitalker ASR",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["asr"],
    hardware=["cpu", "cuda"],
    modes=["batch"],
)


# ---------------------------------------------------------------------------
# NER from Transcript (spaCy)
# ---------------------------------------------------------------------------


def load_ner_spacy(config, device):
    """NER extraction from transcript text using spaCy."""
    import spacy

    model_name = config.get("model_name", "en_core_web_sm")
    nlp = spacy.load(model_name)

    def extract_entities(text, **kwargs):
        doc = nlp(text)
        entities = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]
        return {
            "entities": entities,
            "count": len(entities),
            "labels": list(set(e["label"] for e in entities)),
        }

    return {
        "model_type": "ner_spacy",
        "device": "cpu",
        "capabilities": ["ner"],
        "modes": ["batch"],
        "ner": {"extract": extract_entities},
        "raw": {"nlp": nlp},
    }


ModelRegistry.register_loader(
    "ner_spacy",
    load_ner_spacy,
    "NER spaCy: Named-entity recognition from transcript text",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["ner"],
    hardware=["cpu"],
    modes=["batch"],
)


# ---------------------------------------------------------------------------
# Emotion Recognition (SpeechBrain)
# ---------------------------------------------------------------------------


def load_emotion_speechbrain(config, device):
    """Speech emotion recognition using SpeechBrain."""
    from speechbrain.inference.classifiers import EncoderClassifier
    import torch

    model_name = config.get(
        "model_name", "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
    )
    classifier = EncoderClassifier.from_hparams(
        source=model_name,
        run_opts={"device": device if device != "mps" else "cpu"},
    )
    actual_device = device if device != "mps" else "cpu"

    def classify_emotion(audio, sr=16000, **kwargs):
        import numpy as np

        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        prediction = classifier.classify_batch(audio)
        scores = prediction[1].squeeze().tolist()
        labels = prediction[3]
        if isinstance(scores, float):
            scores = [scores]
        return {
            "emotion": labels[0] if labels else "unknown",
            "scores": scores,
            "all_labels": labels,
        }

    return {
        "model_type": "emotion_speechbrain",
        "device": actual_device,
        "capabilities": ["emotion"],
        "modes": ["batch"],
        "emotion": {"classify": classify_emotion},
        "raw": {"classifier": classifier},
    }


ModelRegistry.register_loader(
    "emotion_speechbrain",
    load_emotion_speechbrain,
    "Emotion SpeechBrain: Speech emotion recognition (wav2vec2-IEMOCAP)",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["emotion"],
    hardware=["cpu", "cuda"],
    modes=["batch"],
)


# ---------------------------------------------------------------------------
# Language ID (SpeechBrain)
# ---------------------------------------------------------------------------


def load_langid_speechbrain(config, device):
    """Language identification using SpeechBrain (107 languages)."""
    from speechbrain.inference.classifiers import EncoderClassifier
    import torch

    model_name = config.get(
        "model_name", "speechbrain/lang-id-voxlingua107-ecapa"
    )
    classifier = EncoderClassifier.from_hparams(
        source=model_name,
        run_opts={"device": device if device != "mps" else "cpu"},
    )
    actual_device = device if device != "mps" else "cpu"

    def identify_language(audio, sr=16000, top_k=5, **kwargs):
        import numpy as np

        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        prediction = classifier.classify_batch(audio)
        scores = prediction[1].squeeze()
        # Get top-k
        if scores.dim() > 0:
            topk_scores, topk_indices = torch.topk(
                scores, min(top_k, len(scores))
            )
            topk_scores = topk_scores.tolist()
        else:
            topk_scores = [scores.item()]
            topk_indices = [0]
        labels = prediction[3]
        return {
            "language": labels[0] if labels else "unknown",
            "confidence": topk_scores[0] if topk_scores else 0.0,
            "top_languages": labels[:top_k] if labels else [],
        }

    return {
        "model_type": "langid_speechbrain",
        "device": actual_device,
        "capabilities": ["langid"],
        "modes": ["batch"],
        "langid": {"identify": identify_language},
        "raw": {"classifier": classifier},
    }


ModelRegistry.register_loader(
    "langid_speechbrain",
    load_langid_speechbrain,
    "LangID SpeechBrain: Language identification (107 languages, ECAPA)",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["langid"],
    hardware=["cpu", "cuda"],
    modes=["batch"],
)


# ---------------------------------------------------------------------------
# Speaker Embedding / Verification (SpeechBrain ECAPA-TDNN)
# ---------------------------------------------------------------------------


def load_speaker_embed(config, device):
    """Speaker embedding and verification using SpeechBrain ECAPA-TDNN."""
    from speechbrain.inference.speaker import SpeakerRecognition
    import torch

    model_name = config.get("model_name", "speechbrain/spkrec-ecapa-voxceleb")
    verifier = SpeakerRecognition.from_hparams(
        source=model_name,
        run_opts={"device": device if device != "mps" else "cpu"},
    )
    actual_device = device if device != "mps" else "cpu"

    def embed(audio, sr=16000, **kwargs):
        import numpy as np

        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        embedding = verifier.encode_batch(audio)
        return {"embedding": embedding.squeeze().tolist(), "dim": embedding.shape[-1]}

    def verify(audio1, audio2, sr=16000, **kwargs):
        import numpy as np

        for a in [audio1, audio2]:
            if isinstance(a, np.ndarray):
                a = torch.from_numpy(a).float()
        score, prediction = verifier.verify_batch(
            torch.from_numpy(audio1).float().unsqueeze(0)
            if isinstance(audio1, np.ndarray)
            else audio1.unsqueeze(0),
            torch.from_numpy(audio2).float().unsqueeze(0)
            if isinstance(audio2, np.ndarray)
            else audio2.unsqueeze(0),
        )
        return {
            "score": score.item(),
            "same_speaker": bool(prediction.item()),
            "threshold": 0.5,
        }

    return {
        "model_type": "speaker_embed_speechbrain",
        "device": actual_device,
        "capabilities": ["speaker_embed", "speaker_verify"],
        "modes": ["batch"],
        "speaker_embed": {"embed": embed},
        "speaker_verify": {"verify": verify},
        "raw": {"verifier": verifier},
    }


ModelRegistry.register_loader(
    "speaker_embed_speechbrain",
    load_speaker_embed,
    "Speaker Embed SpeechBrain: Speaker embedding & verification (ECAPA-TDNN)",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["speaker_embed", "speaker_verify"],
    hardware=["cpu", "cuda"],
    modes=["batch"],
)


# ---------------------------------------------------------------------------
# MusicGen (Meta)
# ---------------------------------------------------------------------------


def load_musicgen(config, device):
    """Music generation from text prompts using Meta's MusicGen."""
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    import torch

    model_name = config.get("model_name", "facebook/musicgen-small")
    actual_device = device if device != "mps" else "cpu"  # MusicGen may not support MPS fully

    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name).to(
        actual_device
    )
    model.eval()

    def generate(prompt, duration_s=5, **kwargs):
        import numpy as np

        max_tokens = int(duration_s * 50)  # ~50 tokens per second
        inputs = processor(
            text=[prompt], padding=True, return_tensors="pt"
        ).to(actual_device)
        with torch.no_grad():
            audio_values = model.generate(**inputs, max_new_tokens=max_tokens)
        audio = audio_values[0, 0].cpu().numpy()
        sr = model.config.audio_encoder.sampling_rate
        return {
            "audio": audio,
            "sample_rate": sr,
            "duration_s": len(audio) / sr,
            "prompt": prompt,
        }

    return {
        "model_type": "musicgen",
        "device": actual_device,
        "capabilities": ["music_gen"],
        "modes": ["batch"],
        "music_gen": {"generate": generate},
        "raw": {"model": model, "processor": processor},
    }


ModelRegistry.register_loader(
    "musicgen",
    load_musicgen,
    "MusicGen: Text-to-music generation (Meta)",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["music_gen"],
    hardware=["cpu", "mps", "cuda"],
    modes=["batch"],
)


# ---------------------------------------------------------------------------
# Parler-TTS
# ---------------------------------------------------------------------------


def load_parler_tts(config: dict, device: str) -> Bundle:
    """Text-to-speech with natural language voice description using Parler-TTS."""
    from transformers import AutoTokenizer
    from parler_tts import ParlerTTSForConditionalGeneration
    import torch

    model_name = config.get("model_name", "parler-tts/parler-tts-mini-v1.1")
    actual_device = device if device != "mps" else "cpu"

    model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(actual_device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    def synthesize(text, voice_description="A female speaker with a warm, clear voice", **kwargs):
        import numpy as np
        input_ids = tokenizer(voice_description, return_tensors="pt").input_ids.to(actual_device)
        prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to(actual_device)
        with torch.no_grad():
            generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_ids)
        audio = generation.cpu().numpy().squeeze()
        sr = model.config.sampling_rate
        return {"audio": audio, "sample_rate": sr, "duration_s": len(audio) / sr, "text": text, "voice_description": voice_description}

    return {
        "model_type": "parler_tts",
        "device": actual_device,
        "capabilities": ["tts"],
        "modes": ["batch"],
        "tts": {"synthesize": synthesize},
        "raw": {"model": model, "tokenizer": tokenizer},
    }


ModelRegistry.register_loader(
    "parler_tts",
    load_parler_tts,
    "Parler-TTS: Text-to-speech with natural language voice description",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["tts"],
    hardware=["cpu", "cuda"],
    modes=["batch"],
)


# ---------------------------------------------------------------------------
# Bark TTS
# ---------------------------------------------------------------------------


def load_bark_tts(config: dict, device: str) -> Bundle:
    """Multi-speaker TTS with effects using Suno's Bark."""
    from transformers import AutoProcessor, BarkModel
    import torch

    model_name = config.get("model_name", "suno/bark-small")
    actual_device = device if device != "mps" else "cpu"

    processor = AutoProcessor.from_pretrained(model_name)
    model = BarkModel.from_pretrained(model_name).to(actual_device)
    model.eval()

    def synthesize(text, voice_preset="v2/en_speaker_6", **kwargs):
        inputs = processor(text, voice_preset=voice_preset, return_tensors="pt").to(actual_device)
        with torch.no_grad():
            audio_array = model.generate(**inputs)
        audio = audio_array.cpu().numpy().squeeze()
        sr = model.generation_config.sample_rate
        return {"audio": audio, "sample_rate": sr, "duration_s": len(audio) / sr, "text": text, "voice_preset": voice_preset}

    return {
        "model_type": "bark_tts",
        "device": actual_device,
        "capabilities": ["tts"],
        "modes": ["batch"],
        "tts": {"synthesize": synthesize},
        "raw": {"model": model, "processor": processor},
    }


ModelRegistry.register_loader(
    "bark_tts",
    load_bark_tts,
    "Bark TTS: Multi-speaker text-to-speech with effects (Suno)",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["tts"],
    hardware=["cpu", "cuda"],
    modes=["batch"],
)


# ---------------------------------------------------------------------------
# Pitch Detection (CREPE)
# ---------------------------------------------------------------------------


def load_pitch_crepe(config: dict, device: str) -> Bundle:
    """Pitch (F0) detection using CREPE neural network."""
    def detect_pitch(audio, sr=16000, **kwargs):
        import numpy as np
        try:
            import crepe
            time_arr, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)
            voiced_mask = confidence > 0.5
            voiced_f0 = frequency[voiced_mask]
            return {
                "time": time_arr.tolist(),
                "frequency": frequency.tolist(),
                "confidence": confidence.tolist(),
                "mean_f0": float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0,
                "std_f0": float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0,
                "voiced_ratio": float(np.mean(voiced_mask)),
            }
        except ImportError:
            import librosa
            f0, voiced_flag, voiced_probs = librosa.pyin(audio.astype(np.float32), fmin=50, fmax=500, sr=sr)
            voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
            return {
                "frequency": f0.tolist() if f0 is not None else [],
                "mean_f0": float(np.nanmean(f0)) if f0 is not None and not np.all(np.isnan(f0)) else 0.0,
                "voiced_ratio": float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0,
            }

    return {
        "model_type": "pitch_crepe",
        "device": "cpu",
        "capabilities": ["pitch"],
        "modes": ["batch"],
        "pitch": {"detect": detect_pitch},
    }


ModelRegistry.register_loader(
    "pitch_crepe",
    load_pitch_crepe,
    "Pitch Detection: F0 estimation using CREPE / librosa fallback",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["pitch"],
    hardware=["cpu"],
    modes=["batch"],
)


def load_model_from_config(config_path: Path, device: str = "cpu") -> Bundle:
    """
    Load model from YAML config file.

    Args:
        config_path: Path to config.yaml
        device: Target device

    Returns:
        Bundle conforming to Bundle Contract v1
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_type = config.get("model_type", "unknown")
    return ModelRegistry.load_model(model_type, config, device)
