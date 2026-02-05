"""
Model registry for loading different models with consistent interface.
Supports LFM2.5-Audio, Whisper, SeamlessM4T, Distil-Whisper, Whisper.cpp, etc.

All loaders MUST return Bundle Contract v1 (see contracts.py).
"""

from __future__ import annotations
import hashlib
from typing import Dict, Any, Optional, Callable, List, cast
from pathlib import Path
import logging
from datetime import datetime
from enum import Enum
import torch

from .contracts import Bundle, validate_bundle, ASRResult

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model lifecycle status."""
    EXPERIMENTAL = "experimental"
    CANDIDATE = "candidate"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


class ModelRegistry:
    """Central registry for model loading and configuration."""

    _models: Dict[str, Bundle] = {}
    _loaders: Dict[str, Dict[str, Any]] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register_loader(cls,
                       model_type: str,
                       loader_func: Callable,
                       description: str = "",
                       status: ModelStatus = ModelStatus.EXPERIMENTAL,
                       version: str = "1.0.0",
                       capabilities: Optional[List[str]] = None,
                       hardware: Optional[List[str]] = None,
                       modes: Optional[List[str]] = None,
                       expected_baseline: Optional[Dict[str, float]] = None):
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

        cls._loaders[model_type] = {
            'loader': loader_func,
            'description': description
        }

        cls._metadata[model_type] = {
            'status': status.value,
            'version': version,
            'capabilities': capabilities,
            'hardware': hardware,
            'modes': modes,
            'date_registered': datetime.now().isoformat(),
            'expected_baseline': expected_baseline or {},
            'observed_baseline': {},  # Only updated from run artifacts
            'hash': hashlib.md5(f"{model_type}:{version}".encode()).hexdigest()[:8]
        }

        logger.debug(f"Registered loader: {model_type} v{version} ({status.value}) caps={capabilities} - {description}")

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
    def update_observed_baseline(cls, model_type: str, metrics: Dict[str, float]):
        """Update observed baseline from actual run artifacts."""
        if model_type in cls._metadata:
            cls._metadata[model_type]['observed_baseline'].update(metrics)
            cls._metadata[model_type]['date_updated'] = datetime.now().isoformat()
            logger.info(f"Updated observed baseline for {model_type}")

    @classmethod
    def list_models_by_status(cls, status: ModelStatus) -> list:
        """List models by status."""
        return [model for model, meta in cls._metadata.items()
                if meta['status'] == status.value]

    @classmethod
    def list_models_by_capability(cls, capability: str) -> list:
        """List models that have a specific capability."""
        return [model for model, meta in cls._metadata.items()
                if capability in meta.get('capabilities', [])]

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
            ModelStatus.EXPERIMENTAL.value: 0,
            ModelStatus.CANDIDATE.value: 1,
            ModelStatus.PRODUCTION.value: 2,
            ModelStatus.DEPRECATED.value: -1
        }

        current_status = metadata.get('status', 'experimental')
        current_level = status_hierarchy.get(current_status, -1)
        required_level = status_hierarchy.get(required_status.value, 2)

        return current_level >= required_level

    @classmethod
    def load_model(cls,
                   model_type: str,
                   config: Dict[str, Any],
                   device: str = 'cpu') -> Bundle:
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
        loader_func = loader_info['loader']

        logger.info(f"Loading {model_type} on {device}...")
        bundle = loader_func(config, device)
        
        # Enforce Bundle Contract v1
        validate_bundle(bundle, model_type)
        
        logger.info(f"✓ Loaded {model_type} with capabilities: {bundle.get('capabilities', [])}")

        cls._models[model_type] = bundle
        return bundle

    @classmethod
    def get_model(cls, model_type: str) -> Optional[Bundle]:
        """Get previously loaded model bundle."""
        return cls._models.get(model_type)

    @classmethod
    def list_models(cls) -> list:
        """List registered model types."""
        return list(cls._loaders.keys())


# =============================================================================
# Model Loader Functions - All must return Bundle Contract v1
# =============================================================================

def load_lfm2_5_audio(config: Dict[str, Any], device: str) -> Bundle:
    """
    Load LFM2.5-Audio model with Bundle Contract v1.
    """
    try:
        from liquid_audio import LFM2AudioModel, LFM2AudioProcessor
        import torch

        model_name = config.get('model_name', 'LiquidAI/LFM2.5-Audio-1.5B')

        # Device selection: MPS > CPU (avoid CUDA due to vendor bug)
        if device == 'cuda':
            actual_device = 'cpu'
            logger.warning(f"LFM2.5-Audio: CUDA requested but using CPU due to vendor CUDA bug")
        elif device == 'mps':
            actual_device = 'mps'
            logger.info(f"LFM2.5-Audio: Using MPS (Apple Silicon) acceleration")
        else:
            actual_device = 'cpu'
            logger.info(f"LFM2.5-Audio: Using CPU")

        model = LFM2AudioModel.from_pretrained(model_name, device=actual_device).eval()
        
        try:
            processor = LFM2AudioProcessor.from_pretrained(model_name, device='cpu')
            if actual_device != 'cpu':
                try:
                    processor = processor.to(actual_device)
                    logger.info(f"✓ Processor moved to {actual_device}")
                except Exception:
                    logger.warning(f"Could not move processor to {actual_device}, keeping on CPU")

            # CRITICAL FIX: Manually initialize audio detokenizer to avoid .cuda() hardcode in library
            # The library's processor.audio_detokenizer property calls .cuda() which fails on MPS
            if processor._audio_detokenizer is None and processor.detokenizer_path:
                try:
                    from liquid_audio.processor import Lfm2Config, LFM2AudioDetokenizer
                    from safetensors.torch import load_file
                    
                    logger.info("Initializing audio detokenizer manually for MPS support...")
                    
                    detok_config_path = Path(processor.detokenizer_path) / "config.json"
                    detok_config = Lfm2Config.from_pretrained(detok_config_path)

                    # Fix layer types mismatch (copied from library)
                    def rename_layer(layer):
                        if layer in ["conv", "full_attention"]: return layer
                        if layer == "sliding_attention": return "full_attention"
                        raise ValueError(f"Unknown layer: {layer}")

                    if isinstance(detok_config.layer_types, list):
                        detok_config.layer_types = [rename_layer(l) for l in detok_config.layer_types]

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
            raise RuntimeError(f"LFM2.5-Audio processor loading failed: {e}")

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
            max_tokens = kwargs.get('max_new_tokens', 512)
            
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
            voice = kwargs.get('voice', 'US female')
            
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
            max_tokens = kwargs.get('max_new_tokens', 512)
            
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
            
            return {"audio": audio, "sample_rate": sr, "meta": {"model": model_name, "voice": voice}}

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
            max_tokens = kwargs.get('max_new_tokens', 768)
            
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
                    response_text = response_text.replace("<|text_end|>", "").replace("<|im_end|>", "").strip()

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
                    "meta": {"model": model_name}
                }
                
            except Exception as e:
                logger.error(f"V2V generation failed: {e}")
                return {"audio": np.zeros(0), "response_text": "", "meta": {"error": str(e)}}

        return {
            'model_type': 'lfm2_5_audio',
            'device': actual_device,
            'capabilities': ['asr', 'tts', 'chat', 'v2v'],
            'modes': ['batch'],
            'asr': {'transcribe': transcribe},
            'tts': {'synthesize': synthesize},
            'chat': {'respond': run_v2v_turn}, # Use same function for chat
            'v2v': {'run_v2v_turn': run_v2v_turn},
            'raw': {'model': model, 'processor': processor}
        }

    except Exception as e:
        raise RuntimeError(f"LFM2.5-Audio loading failed: {e}")


def load_whisper(config: Dict[str, Any], device: str) -> Bundle:
    """Load Whisper model with Bundle Contract v1."""
    try:
        import whisper
        import torch # Added for internal use

        model_name = config.get('model_name', 'large-v3')
        model = whisper.load_model(model_name, device=device)

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
                "meta": result
            }

        def align(audio, sr=16000, **kwargs):
            """Align audio (same as transcribe + segments)."""
            return transcribe(audio, sr, **kwargs)

        return {
            'model_type': 'whisper',
            'device': device,
            'capabilities': ['asr', 'alignment'],
            'modes': ['batch'],
            'asr': {'transcribe': transcribe},
            'alignment': {'align': align},
            'raw': {'model': model}
        }

    except ImportError:
        raise ImportError("whisper package not installed. Install with: uv add openai-whisper")


def load_faster_whisper(config: Dict[str, Any], device: str) -> Bundle:
    """Load Faster-Whisper model with Bundle Contract v1."""
    try:
        from faster_whisper import WhisperModel
        import torch # Added for consistency if needed, though fw uses numpy

        model_name = config.get('model_name', 'large-v3')
        compute_type = config.get('inference', {}).get('compute_type', 'float16')

        # Map device names for faster-whisper
        if device == 'mps':
            device_fw = 'cpu'  # faster-whisper doesn't support MPS directly
        elif device == 'cuda':
            device_fw = 'cuda'
        else:
            device_fw = 'cpu'

        # Force int8/float32 on CPU if float16 requested (which is default but fails on many CPUs)
        if device_fw == 'cpu' and compute_type == 'float16':
            compute_type = 'int8'
            # logger.info("Forced compute_type='int8' for CPU execution")

        model = WhisperModel(
            model_name,
            device=device_fw,
            compute_type=compute_type
        )

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
                "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segments_list],
                "language": info.language if info else "",
                "meta": {"language_probability": info.language_probability if info else 0}
            }

        return {
            'model_type': 'faster_whisper',
            'device': device,
            'capabilities': ['asr'],
            'modes': ['batch', 'streaming'],
            'asr': {'transcribe': transcribe},
            'raw': {'model': model, 'device_fw': device_fw}
        }

    except ImportError:
        raise ImportError("faster-whisper package not installed. Install with: uv add faster-whisper")


def load_seamlessm4t(config: Dict[str, Any], device: str) -> Bundle:
    """Load SeamlessM4T model with Bundle Contract v1."""
    try:
        from transformers import SeamlessM4TForSpeechToText, AutoProcessor
        import torch

        model_name = config.get('model_name', 'facebook/seamless-m4t-v2-large')
        processor = AutoProcessor.from_pretrained(model_name)
        model = SeamlessM4TForSpeechToText.from_pretrained(model_name)

        if device != 'cpu':
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
            if device != 'cpu':
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Filter out kwargs that model.generate() doesn't accept
            generate_kwargs = {k: v for k, v in kwargs.items() if k not in ('language', 'lang')}
            
            with torch.no_grad():
                outputs = model.generate(**inputs, tgt_lang="eng", **generate_kwargs)
            
            text = processor.decode(outputs[0], skip_special_tokens=True)
            return {"text": text, "segments": [], "meta": {"model": model_name}}

        return {
            'model_type': 'seamlessm4t',
            'device': device,
            'capabilities': ['asr', 'mt'],
            'modes': ['batch'],
            'asr': {'transcribe': transcribe},
            'mt': {'translate': transcribe},  # Same function for now
            'raw': {'model': model, 'processor': processor}
        }

    except ImportError:
        raise ImportError("transformers package not installed. Install with: uv add transformers")


def load_distil_whisper(config: Dict[str, Any], device: str) -> Bundle:
    """
    Load Distil-Whisper via transformers with Bundle Contract v1.
    Uses direct generate() instead of pipeline for MPS/CUDA parity.
    """
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        import torch

        model_id = config.get("model_name", "distil-whisper/distil-large-v3")

        # dtype selection
        dtype_cfg = config.get("inference", {}).get("dtype", None)
        if dtype_cfg == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float16 if device != "cpu" else torch.float32

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).eval()

        if device != "cpu":
            model = model.to(device)

        def transcribe(audio, sr=16000, **kwargs):
            """Transcribe audio using Distil-Whisper."""
            import numpy as np
            if isinstance(audio, torch.Tensor):
                audio_np = audio.detach().cpu().numpy()
            else:
                audio_np = audio

            inputs = processor(
                audio_np,
                sampling_rate=sr,
                return_tensors="pt"
            )
            # Cast to model dtype when moving to device
            inputs = {k: v.to(device=device, dtype=torch_dtype if v.is_floating_point() else None) for k, v in inputs.items()}

            gen_kwargs = {
                "max_new_tokens": kwargs.get("max_new_tokens", 256),
                "num_beams": kwargs.get("num_beams", 1),
            }

            with torch.no_grad():
                predicted_ids = model.generate(**inputs, **gen_kwargs)

            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return {"text": text, "segments": [], "meta": {"model": model_id}}

        return {
            "model_type": "distil_whisper",
            "device": device,
            "capabilities": ["asr"],
            "modes": ["batch"],
            "asr": {"transcribe": transcribe},
            "raw": {"model": model, "processor": processor}
        }

    except ImportError as e:
        raise ImportError("transformers not installed. Install with: uv add transformers accelerate") from e


def load_whisper_cpp(config: Dict[str, Any], device: str) -> Bundle:
    """
    Whisper.cpp CLI adapter with Bundle Contract v1.
    Expects whisper.cpp binary built and accessible.
    
    Provides both transcribe() and transcribe_path() - runner uses transcribe().
    """
    import subprocess
    import shutil
    import tempfile
    import soundfile as sf
    import numpy as np
    
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
            "-m", model_path,
            "-f", str(audio_path),
            "-l", lang,
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
        import torch # Need torch to check if tensor
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
        "raw": {"bin_path": bin_path, "model_path": model_path}
    }


def load_silero_vad(config: Dict[str, Any], device: str) -> Bundle:
    """Load Silero VAD model with Bundle Contract v1."""
    try:
        import torch

        # Load from Torch Hub
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False,
                                      onnx=False,
                                      trust_repo=True)
        
        (get_speech_timestamps, _, _, _, _) = utils
        
        if device != 'cpu':
            model = model.to(device)
        model.eval()

        def detect(audio, sr=16000, **kwargs):
            """Detect speech segments."""
            # Silero expects tensor
            import numpy as np
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)
            
            if audio.dim() > 1:
                audio = audio.squeeze() # Ensure 1D
                
            if device != 'cpu':
                audio = audio.to(device)
                
            # Run detection
            timestamps = get_speech_timestamps(audio, model, sampling_rate=sr, **kwargs)
            
            return {
                "segments": timestamps, # List of dicts {start: int, end: int} in samples
                "meta": {"model": "silero_vad"}
            }

        return {
            "model_type": "silero_vad",
            "device": device,
            "capabilities": ["vad"],
            "modes": ["batch"],
            "vad": {"detect": detect},
            "raw": {"model": model, "utils": utils}
        }


    except Exception as e:
        raise RuntimeError(f"Silero VAD loading failed: {e}")


def load_pyannote_diarization(config: Dict[str, Any], device: str) -> Bundle:
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
        use_auth_token = config.get("auth_token", True)
        
        logger.info(f"Loading pyannote pipeline: {model_name}")
        
        # Load pipeline
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=use_auth_token)
        
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
                audio = audio.unsqueeze(0) # Add channel dim [1, T]

            # Ensure float32
            if audio.dtype != torch.float32:
                audio = audio.float()
                
            file = {"waveform": audio, "sample_rate": sr}
            
            # Run inference
            diarization = pipeline(file, **kwargs)
            
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
                
            return {
                "segments": segments,
                "num_speakers": len(set(s['speaker'] for s in segments)),
                "meta": {"model": model_name}
            }

        return {
            "model_type": "pyannote_diarization",
            "device": device,
            "capabilities": ["diarization"],
            "modes": ["batch"],
            "diarization": {"diarize": diarize},
            "raw": {"pipeline": pipeline}
        }

    except Exception as e:
        raise RuntimeError(f"Pyannote loading failed: {e}")


def load_heuristic_diarizer(config: Dict[str, Any], device: str) -> Bundle:
    """
    Load Heuristic Diarizer (Silero VAD + Single Speaker assumption).
    Fallback for when pyannote is not available.
    """
    try:
        import torch
        # Reuse Silero VAD loading logic
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False,
                                      trust_repo=True)
        (get_speech_timestamps, _, _, _, _) = utils
        
        if device != 'cpu':
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
                
            if device != 'cpu':
                audio = audio.to(device)
            
            timestamps = get_speech_timestamps(audio, model, sampling_rate=sr, return_seconds=True)
            
            # Convert to diarization format (add speaker label)
            segments = []
            for ts in timestamps:
                segments.append({
                    "start": ts['start'],
                    "end": ts['end'],
                    "speaker": "SPEAKER_00"
                })
                
            return {
                "segments": segments,
                "num_speakers": 1 if segments else 0,
                "meta": {"model": "heuristic_diarizer", "backend": "silero_vad"}
            }

        return {
            "model_type": "heuristic_diarizer",
            "device": device,
            "capabilities": ["diarization"],
            "modes": ["batch"],
            "diarization": {"diarize": diarize},
            "raw": {"model": model}
        }

    except Exception as e:
        raise RuntimeError(f"Heuristic Diarizer loading failed: {e}")



# =============================================================================
# Register default loaders
# =============================================================================

ModelRegistry.register_loader(
    'lfm2_5_audio',
    load_lfm2_5_audio,
    'LiquidAI LFM-2.5-Audio model for ASR, TTS, and conversation',
    status=ModelStatus.CANDIDATE,
    version="2.5.0",
    capabilities=["asr", "tts", "chat"],
    hardware=["cpu", "mps"],
    modes=["batch"]
)

ModelRegistry.register_loader(
    'whisper',
    load_whisper,
    'OpenAI Whisper model for ASR',
    status=ModelStatus.PRODUCTION,
    version="3.0.0",
    capabilities=["asr"],
    hardware=["cpu", "mps", "cuda"],
    modes=["batch"]
)

ModelRegistry.register_loader(
    'faster_whisper',
    load_faster_whisper,
    'Optimized Whisper implementation for faster inference (guillaumekln/faster-whisper)',
    status=ModelStatus.PRODUCTION,
    version="1.0.0",
    capabilities=["asr"],
    hardware=["cpu", "cuda"],
    modes=["batch", "streaming"]
)

ModelRegistry.register_loader(
    'seamlessm4t',
    load_seamlessm4t,
    'Meta SeamlessM4T for multi-modal speech translation',
    status=ModelStatus.EXPERIMENTAL,
    version="2.0.0",
    capabilities=["asr", "mt"],
    hardware=["cpu", "mps", "cuda"],
    modes=["batch"]
)

ModelRegistry.register_loader(
    'distil_whisper',
    load_distil_whisper,
    'Distil-Whisper: 6x faster Whisper with minimal accuracy loss',
    status=ModelStatus.EXPERIMENTAL,
    version="3.0.0",
    capabilities=["asr"],
    hardware=["cpu", "mps", "cuda"],
    modes=["batch"]
)

ModelRegistry.register_loader(
    'pyannote_diarization',
    load_pyannote_diarization,
    'Pyannote.audio Speaker Diarization pipeline',
    status=ModelStatus.PRODUCTION,
    version="3.1.0",
    capabilities=["diarization"],
    hardware=["cpu", "cuda"],
    modes=["batch"]
)

ModelRegistry.register_loader(
    'heuristic_diarization',
    load_heuristic_diarizer,
    'Heuristic Diarizer (VAD + 1-speaker assumption)',
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["diarization"],
    hardware=["cpu", "cuda", "mps"],
    modes=["batch"]
)

ModelRegistry.register_loader(
    'whisper_cpp',
    load_whisper_cpp,
    'whisper.cpp: edge-friendly C++ inference backend',
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["asr"],
    hardware=["cpu"],
    modes=["cli"]
)


ModelRegistry.register_loader(
    'silero_vad',
    load_silero_vad,
    'Silero VAD: Production-grade voice activity detection',
    status=ModelStatus.PRODUCTION,
    version="4.0.0",
    capabilities=["vad"],
    hardware=["cpu", "mps", "cuda"],
    modes=["batch"]
)



def load_model_from_config(config_path: Path, device: str = 'cpu') -> Bundle:
    """
    Load model from YAML config file.

    Args:
        config_path: Path to config.yaml
        device: Target device

    Returns:
        Bundle conforming to Bundle Contract v1
    """
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_type = config.get('model_type', 'unknown')
    return ModelRegistry.load_model(model_type, config, device)