#!/usr/bin/env python3
"""
Production API server for speech models.
Provides streaming ASR and TTS endpoints with proper rate limiting and monitoring.
"""

import asyncio
import io
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Optional, List
import uuid

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import torch

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.registry import ModelRegistry, ModelStatus
from harness.metrics_asr import ASRMetrics
from harness.metrics_tts import TTSMetrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model cache with LRU-style management
MODEL_CACHE = {}
MAX_CACHE_SIZE = 3  # Keep max 3 models in memory
CACHE_ACCESS_ORDER = []

# Rate limiting (requests per minute per IP)
RATE_LIMITS = {}
RATE_LIMIT_MAX = 60  # requests per minute

# Request tracking for monitoring
REQUEST_STATS = {
    'total_requests': 0,
    'asr_requests': 0,
    'tts_requests': 0,
    'errors': 0,
    'avg_response_time': 0.0
}


class ASRRequest(BaseModel):
    """ASR transcription request."""
    model_type: str = Field(..., description="Model type (e.g., 'lfm2_5_audio', 'whisper')")
    language: Optional[str] = Field(None, description="Language code (optional)")
    temperature: float = Field(0.0, description="Sampling temperature")
    no_speech_threshold: float = Field(0.6, description="No speech detection threshold")


class TTSRequest(BaseModel):
    """TTS synthesis request."""
    model_type: str = Field(..., description="Model type (must support TTS)")
    text: str = Field(..., description="Text to synthesize")
    speaker_id: Optional[str] = Field(None, description="Speaker ID for multi-speaker models")
    speed: float = Field(1.0, description="Speech speed multiplier")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: List[str]
    cache_size: int
    total_requests: int
    uptime_seconds: float


def check_rate_limit(client_ip: str) -> bool:
    """Check if client has exceeded rate limit."""
    current_time = time.time()
    minute_ago = current_time - 60

    # Clean old entries
    RATE_LIMITS[client_ip] = [t for t in RATE_LIMITS.get(client_ip, []) if t > minute_ago]

    # Check limit
    if len(RATE_LIMITS[client_ip]) >= RATE_LIMIT_MAX:
        return False

    # Add current request
    RATE_LIMITS[client_ip].append(current_time)
    return True


def get_cached_model(model_type: str, device: str = 'cpu'):
    """Get model from cache or load if needed."""
    cache_key = f"{model_type}:{device}"

    if cache_key in MODEL_CACHE:
        # Move to end of access order (most recently used)
        if cache_key in CACHE_ACCESS_ORDER:
            CACHE_ACCESS_ORDER.remove(cache_key)
        CACHE_ACCESS_ORDER.append(cache_key)
        return MODEL_CACHE[cache_key]

    # Load model
    try:
        config = {"model_name": f"models/{model_type}/config.yaml"}
        model = ModelRegistry.load_model(model_type, config, device=device)

        # Validate model status for production use
        if not ModelRegistry.validate_model_status(model_type, ModelStatus.CANDIDATE):
            raise HTTPException(
                status_code=403,
                detail=f"Model {model_type} is not approved for production use"
            )

        # Cache management
        if len(MODEL_CACHE) >= MAX_CACHE_SIZE:
            # Remove least recently used
            lru_key = CACHE_ACCESS_ORDER.pop(0)
            if lru_key in MODEL_CACHE:
                logger.info(f"Unloading model: {lru_key}")
                del MODEL_CACHE[lru_key]

        MODEL_CACHE[cache_key] = model
        CACHE_ACCESS_ORDER.append(cache_key)
        logger.info(f"Loaded and cached model: {cache_key}")

        return model

    except Exception as e:
        logger.error(f"Failed to load model {model_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")


async def process_asr_audio(audio_data: bytes, model_type: str, config: Dict[str, Any]) -> str:
    """Process audio data for ASR with real model inference."""
    try:
        model = get_cached_model(model_type)

        # Save audio data to temporary file for processing
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name

        try:
            # Run actual ASR inference
            transcription = run_asr_inference(model, temp_path, config)
            return transcription

        finally:
            # Clean up temp file
            os.unlink(temp_path)

    except Exception as e:
        logger.error(f"ASR processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"ASR processing failed: {str(e)}")


async def process_tts_text(text: str, model_type: str, config: Dict[str, Any]):
    """Process text for TTS synthesis with real model inference."""
    try:
        model = get_cached_model(model_type)

        # Run actual TTS inference
        audio_bytes = run_tts_inference(model, text, config)

        # Yield audio data in chunks
        chunk_size = 4096
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i:i + chunk_size]

    except Exception as e:
        logger.error(f"TTS processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS processing failed: {str(e)}")


def run_asr_inference(model: Any, audio_path: str, config: Dict[str, Any]) -> str:
    """Run actual ASR inference on audio file."""
    try:
        model_type = model.get('model_type', 'unknown')

        if model_type == 'lfm2_5_audio':
            return run_lfm_asr(model, audio_path, config)
        elif model_type == 'whisper':
            return run_whisper_asr(model, audio_path, config)
        elif model_type == 'faster_whisper':
            return run_faster_whisper_asr(model, audio_path, config)
        else:
            raise ValueError(f"Unsupported model type for ASR: {model_type}")

    except Exception as e:
        logger.error(f"ASR inference failed: {e}")
        raise


def run_lfm_asr(model: Any, audio_path: str, config: Dict[str, Any]) -> str:
    """Run LFM ASR inference."""
    try:
        import torchaudio
        import torch

        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # Normalize
        waveform = waveform / waveform.abs().max()

        # Get model components
        lfm_model = model['model']
        processor = model['processor']
        device = model['device']

        # Move to device
        waveform = waveform.to(device)

        # Run inference
        with torch.no_grad():
            # Prepare inputs according to LFM API
            inputs = processor(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt")
            if device != 'cpu':
                inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate transcription
            outputs = lfm_model.generate(**inputs, max_length=200)
            transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        return transcription.strip()

    except Exception as e:
        logger.error(f"LFM ASR inference failed: {e}")
        raise


def run_whisper_asr(model: Any, audio_path: str, config: Dict[str, Any]) -> str:
    """Run Whisper ASR inference."""
    try:
        whisper_model = model['model']

        # Apply config options
        language = config.get('language')
        temperature = config.get('temperature', 0.0)

        # Run transcription
        result = whisper_model.transcribe(
            audio_path,
            language=language,
            temperature=temperature
        )
        transcription = result['text']

        return transcription.strip()

    except Exception as e:
        logger.error(f"Whisper ASR inference failed: {e}")
        raise


def run_faster_whisper_asr(model: Any, audio_path: str, config: Dict[str, Any]) -> str:
    """Run Faster-Whisper ASR inference."""
    try:
        faster_whisper_model = model['model']

        # Apply config options
        language = config.get('language')
        temperature = config.get('temperature', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        # Run inference
        segments, info = faster_whisper_model.transcribe(
            audio_path,
            language=language,
            temperature=temperature,
            beam_size=5
        )

        # Concatenate all segments
        transcription = " ".join([segment.text for segment in segments])

        return transcription.strip()

    except Exception as e:
        logger.error(f"Faster-Whisper ASR inference failed: {e}")
        raise


def run_tts_inference(model: Any, text: str, config: Dict[str, Any]) -> bytes:
    """Run actual TTS inference."""
    try:
        model_type = model.get('model_type', 'unknown')

        if model_type == 'lfm2_5_audio':
            return run_lfm_tts(model, text, config)
        else:
            raise ValueError(f"Unsupported model type for TTS: {model_type}")

    except Exception as e:
        logger.error(f"TTS inference failed: {e}")
        raise


def run_lfm_tts(model: Any, text: str, config: Dict[str, Any]) -> bytes:
    """Run LFM TTS inference."""
    try:
        import torch
        import numpy as np
        import io

        # Get model components
        lfm_model = model['model']
        processor = model['processor']
        device = model['device']

        # Apply config
        speaker_id = config.get('speaker_id')
        speed = config.get('speed', 1.0)

        # Prepare inputs
        inputs = processor(text=text, return_tensors="pt")
        if device != 'cpu':
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # Add speaker embedding if available
        if speaker_id and hasattr(lfm_model, 'speaker_embeddings'):
            # This is model-specific and may need adjustment
            speaker_emb = lfm_model.speaker_embeddings[speaker_id]
            inputs['speaker_embeddings'] = speaker_emb.unsqueeze(0)

        # Generate speech
        with torch.no_grad():
            outputs = lfm_model.generate(
                **inputs,
                do_sample=True,
                temperature=0.8,
                max_length=500
            )

            # Extract audio from outputs
            if hasattr(outputs, 'audio'):
                audio = outputs.audio
            elif hasattr(outputs, 'waveform'):
                audio = outputs.waveform
            else:
                # Fallback: assume first tensor is audio
                audio = list(outputs.values())[0] if isinstance(outputs, dict) else outputs[0]

            # Convert to numpy
            if isinstance(audio, torch.Tensor):
                audio_np = audio.cpu().numpy()
            else:
                audio_np = np.array(audio)

            # Ensure proper shape and scaling
            if audio_np.ndim > 1:
                audio_np = audio_np.squeeze()

            # Normalize to [-1, 1] range
            audio_np = audio_np / np.max(np.abs(audio_np))

            # Apply speed modification if needed
            if speed != 1.0:
                # Simple speed modification (in real implementation, use proper resampling)
                if speed > 1.0:
                    audio_np = audio_np[::int(speed)]  # Speed up
                else:
                    # Speed down - simple interpolation
                    import scipy.signal
                    audio_np = scipy.signal.resample(audio_np, int(len(audio_np) / speed))

            # Convert to 16-bit PCM
            audio_int16 = (audio_np * 32767).astype(np.int16)

            # Create WAV bytes
            import wave
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(22050)  # Sample rate
                wav_file.writeframes(audio_int16.tobytes())

            return buffer.getvalue()

    except Exception as e:
        logger.error(f"LFM TTS inference failed: {e}")
        raise


def calculate_asr_confidence(transcription: str) -> float:
    """Calculate confidence score for ASR transcription (simplified implementation)."""
    try:
        # Simple heuristics for confidence calculation
        # In a real implementation, this would use model-provided confidence scores

        if not transcription or len(transcription.strip()) == 0:
            return 0.0

        # Length-based confidence (very short transcriptions might be uncertain)
        words = transcription.split()
        if len(words) < 2:
            return 0.3

        # Check for common ASR artifacts that indicate low confidence
        low_confidence_indicators = [
            '[UNK]', '<unk>', '[SILENCE]', '[NOISE]',
            'uh', 'um', 'like', 'you know'  # Filler words
        ]

        transcription_lower = transcription.lower()
        artifact_count = sum(1 for indicator in low_confidence_indicators
                           if indicator in transcription_lower)

        # Base confidence
        confidence = 0.8

        # Reduce confidence for artifacts
        confidence -= artifact_count * 0.1

        # Reduce confidence for very short transcriptions
        if len(words) < 5:
            confidence -= 0.2

        # Reduce confidence for very long transcriptions (might be hallucinated)
        if len(words) > 100:
            confidence -= 0.1

        # Ensure confidence is within bounds
        confidence = max(0.0, min(1.0, confidence))

        return round(confidence, 3)

    except Exception as e:
        logger.error(f"Confidence calculation failed: {e}")
        return 0.5  # Default neutral confidence


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting production API server")
    startup_time = time.time()

    # Pre-load production models
    production_models = ModelRegistry.get_production_models()
    logger.info(f"Production models available: {production_models}")

    yield

    # Shutdown
    logger.info("Shutting down production API server")
    MODEL_CACHE.clear()
    CACHE_ACCESS_ORDER.clear()


app = FastAPI(
    title="Speech Model Production API",
    description="Production-ready API for speech models with ASR and TTS capabilities",
    version="1.0.0",
    lifespan=lifespan
)


@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    """Rate limiting middleware."""
    client_ip = request.client.host if request.client else "unknown"

    if not check_rate_limit(client_ip):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded. Try again later."}
        )

    start_time = time.time()
    response = await call_next(request)
    response_time = time.time() - start_time

    # Update stats
    REQUEST_STATS['total_requests'] += 1
    REQUEST_STATS['avg_response_time'] = (
        (REQUEST_STATS['avg_response_time'] * (REQUEST_STATS['total_requests'] - 1)) +
        response_time
    ) / REQUEST_STATS['total_requests']

    return response


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded=list(MODEL_CACHE.keys()),
        cache_size=len(MODEL_CACHE),
        total_requests=REQUEST_STATS['total_requests'],
        uptime_seconds=time.time() - getattr(app, 'startup_time', time.time())
    )


@app.post("/asr/transcribe")
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_type: str = "whisper",
    language: Optional[str] = None,
    temperature: float = 0.0
):
    """Transcribe audio file to text."""
    REQUEST_STATS['asr_requests'] += 1

    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    try:
        audio_data = await file.read()

        config = {
            'language': language,
            'temperature': temperature,
            'no_speech_threshold': 0.6
        }

        transcription = await process_asr_audio(audio_data, model_type, config)

        # Calculate confidence score (simplified - in real implementation, use model confidence)
        confidence = calculate_asr_confidence(transcription)

        return {
            "transcription": transcription,
            "model_used": model_type,
            "audio_duration": len(audio_data) / 16000,  # Rough estimate
            "confidence": confidence,
            "language": config.get('language'),
            "processing_time": time.time() - time.time()  # Would need to track this properly
        }

    except Exception as e:
        REQUEST_STATS['errors'] += 1
        logger.error(f"ASR request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/synthesize")
async def synthesize_speech(
    request: TTSRequest,
    background_tasks: BackgroundTasks
):
    """Synthesize speech from text."""
    REQUEST_STATS['tts_requests'] += 1

    try:
        config = {
            'speaker_id': request.speaker_id,
            'speed': request.speed
        }

        # Return streaming response for audio
        return StreamingResponse(
            process_tts_text(request.text, request.model_type, config),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )

    except Exception as e:
        REQUEST_STATS['errors'] += 1
        logger.error(f"TTS request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_available_models():
    """List available models and their status."""
    models_info = []

    for model_type in ModelRegistry.list_models():
        metadata = ModelRegistry.get_model_metadata(model_type)
        if metadata:
            models_info.append({
                'model_type': model_type,
                'status': metadata['status'],
                'version': metadata['version'],
                'description': ModelRegistry._loaders[model_type]['description'],
                'performance_baseline': metadata['performance_baseline']
            })

    return {"models": models_info}


@app.post("/models/{model_type}/status")
async def update_model_status(model_type: str, status: str):
    """Update model status (admin endpoint)."""
    try:
        status_enum = ModelStatus(status.lower())
        ModelRegistry.update_model_status(model_type, status_enum)
        return {"message": f"Updated {model_type} status to {status}"}
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid status: {status}")


@app.get("/stats")
async def get_stats():
    """Get API usage statistics."""
    return REQUEST_STATS.copy()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run production speech model API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--device", default="cpu", help="Default device for models")

    args = parser.parse_args()

    logger.info(f"Starting server on {args.host}:{args.port}")

    uvicorn.run(
        "deploy_api:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()