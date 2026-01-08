#!/usr/bin/env python3
"""
Headless ASR runner for production testing.
Usage: python scripts/run_asr.py --model faster_whisper --dataset primary
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add harness to path
harness_path = Path(__file__).parent.parent / 'harness'
sys.path.insert(0, str(harness_path))

from audio_io import AudioLoader, GroundTruthLoader
from registry import ModelRegistry
from timers import PerformanceTimer
from metrics_asr import ASRMetrics
from protocol import RunContract, NormalizationValidator, SegmentationValidator, create_validation_report
import yaml


def load_model_config(model_id: str) -> dict:
    """Load model configuration."""
    # Map model IDs to config paths
    model_configs = {
        'lfm2_5_audio': Path('models/lfm2_5_audio/config.yaml'),
        'whisper': Path('models/whisper/config.yaml'),
        'faster_whisper': Path('models/faster_whisper/config.yaml'),
        'seamlessm4t': Path('models/seamlessm4t/config.yaml'),
    }

    if model_id not in model_configs:
        raise ValueError(f"Unknown model: {model_id}. Available: {list(model_configs.keys())}")

    config_path = model_configs[model_id]
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_dataset_files(dataset: str) -> tuple:
    """Get audio and ground truth files for dataset."""
    datasets = {
        'smoke': {
            'audio': Path('data/audio/SMOKE/conversation_2ppl_10s.wav'),
            'text': Path('data/text/SMOKE/conversation_2ppl_10s.txt')
        },
        'primary': {
            'audio': Path('data/audio/PRIMARY/llm_recording_pranay.wav'),
            'text': Path('data/text/PRIMARY/llm.txt')
        },
        'conversation': {
            'audio': Path('data/audio/PRIMARY/UX_Psychology_From_Miller_s_Law_to_AI.wav'),
            'text': None  # No ground truth for conversation
        }
    }

    if dataset not in datasets:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(datasets.keys())}")

    return datasets[dataset]['audio'], datasets[dataset]['text']


def transcribe_whisper(model_wrapper, audio, sr):
    """Transcribe using Whisper model."""
    model = model_wrapper['model']
    import numpy as np
    # Convert audio to float32 for Whisper compatibility
    audio_float32 = audio.astype(np.float32)
    result = model.transcribe(audio_float32, language='en')
    return result['text'].strip()


def transcribe_faster_whisper(model_wrapper, audio, sr):
    """Transcribe using Faster-Whisper model."""
    model = model_wrapper['model']
    import numpy as np
    # Convert audio to float32 for faster-whisper compatibility
    audio_float32 = audio.astype(np.float32)
    segments, info = model.transcribe(audio_float32, beam_size=5, language='en')

    # Combine all segments
    text_parts = []
    for segment in segments:
        text_parts.append(segment.text)

    return ''.join(text_parts).strip()


def transcribe_seamlessm4t(model_wrapper, audio, sr):
    """Transcribe using SeamlessM4T model with chunking for long audio."""
    model = model_wrapper['model']
    processor = model_wrapper.get('processor')
    import torch
    import numpy as np
    
    # Check audio duration
    duration = len(audio) / sr
    max_chunk_duration = 120  # 2 minutes max per chunk
    
    if duration <= max_chunk_duration:
        # Short audio - process normally
        return _transcribe_seamlessm4t_chunk(model_wrapper, audio, sr)
    else:
        # Long audio - split into chunks
        print(f"Audio too long ({duration:.1f}s), chunking into {max_chunk_duration}s segments...")
        chunks = []
        chunk_samples = int(max_chunk_duration * sr)
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            if len(chunk) < chunk_samples * 0.1:  # Skip very short final chunks
                continue
            chunks.append(chunk)
        
        print(f"Processing {len(chunks)} chunks...")
        transcriptions = []
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            try:
                text = _transcribe_seamlessm4t_chunk(model_wrapper, chunk, sr)
                transcriptions.append(text)
            except Exception as e:
                print(f"Warning: Failed to process chunk {i+1}: {e}")
                transcriptions.append("[CHUNK_FAILED]")
        
        return ' '.join(transcriptions).strip()


def _transcribe_seamlessm4t_chunk(model_wrapper, audio, sr):
    """Transcribe a single chunk using SeamlessM4T model."""
    model = model_wrapper['model']
    processor = model_wrapper.get('processor')
    import torch
    import numpy as np
    
    # Convert audio to float32
    audio_float32 = audio.astype(np.float32)
    
    # Process audio
    if processor:
        inputs = processor(audios=audio_float32, return_tensors='pt')
        # Move to device
        device = model_wrapper.get('device', 'cpu')
        if device != 'cpu':
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            gen_out = model.generate(**inputs, tgt_lang='eng')
        
        # Extract sequences from generate output
        if hasattr(gen_out, "sequences"):       # GenerateOutput
            sequences = gen_out.sequences
        else:                                   # tensor
            sequences = gen_out
        
        # Ensure shape [batch, seq]
        if sequences.dim() == 1:
            sequences = sequences.unsqueeze(0)
        
        # Force integer ids
        sequences = sequences.detach().to("cpu").to(torch.int64)
        
        # Decode
        translated_text = processor.batch_decode(sequences, skip_special_tokens=True)[0]
    else:
        # Fallback if no processor
        translated_text = ""
    
    return translated_text.strip() if translated_text else ""


def transcribe_lfm2_5_audio(model_wrapper, audio, sr):
    """
    Transcribe using LFM2.5-Audio model.
    
    Handles audio format conversions for liquid-audio compatibility:
    - Converts numpy arrays to PyTorch tensors
    - Reshapes 1D audio (samples,) to 2D (channels, samples)
    - Manages device movement for processor and model
    
    Args:
        model_wrapper: Dictionary with 'model' and 'processor' keys
        audio: Audio data (numpy array or torch tensor)
        sr: Sample rate in Hz
    
    Returns:
        Transcribed text string
        
    Raises:
        RuntimeError: If transcription fails for any reason
        
    Note:
        liquid-audio's ChatState.add_audio() requires:
        1. PyTorch tensors (not numpy arrays)
        2. 2D shape with channels: (channels, samples)
        See: LFM_MPS_FIX_SUMMARY.md for detailed explanation.
    """
    from liquid_audio import ChatState
    import torch
    import numpy as np

    model = model_wrapper['model']
    processor = model_wrapper['processor']

    try:
        # Create chat state for ASR
        chat = ChatState(processor)
        chat.new_turn("system")
        chat.add_text("Perform ASR.")
        chat.end_turn()

        # Add audio - liquid-audio expects torch tensor (channels, samples)
        # Convert numpy array to tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio.float()
        
        # Ensure 2D shape (channels, samples)
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
        
        chat.new_turn("user")
        chat.add_audio(audio_tensor, sr)
        chat.end_turn()

        chat.new_turn("assistant")

        # Generate transcription
        text_tokens = []

        for token in model.generate_sequential(**chat, max_new_tokens=512):
            if token.numel() == 1:  # Text token
                text_tokens.append(token)

        # Decode text
        if text_tokens:
            text_tensor = torch.stack(text_tokens, 1)
            text = processor.text.decode(text_tensor[0])
        else:
            text = ""

        return text.strip()

    except Exception as e:
        raise RuntimeError(f"LFM2.5-Audio transcription failed: {e}")


def run_asr_test(model_id: str, dataset: str, device: str = None):
    """Run ASR test and generate results."""
    print(f"=== ASR Test: {model_id} on {dataset} ===")

    # Load model config
    config = load_model_config(model_id)
    print(f"Model: {config['model_name']}")

    # Determine device
    if device is None:
        import torch
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load model
    model_wrapper = ModelRegistry.load_model(config['model_type'], config, device)
    print(f"✓ Model loaded")

    # Load dataset
    audio_path, text_path = get_dataset_files(dataset)
    print(f"Audio: {audio_path}")

    loader = AudioLoader(target_sample_rate=config['audio']['sample_rate'])
    audio, sr, metadata = loader.load_audio(audio_path, config['model_type'])

    # Load ground truth if available
    ground_truth = None
    if text_path and text_path.exists():
        ground_truth = GroundTruthLoader.load_text(text_path)
        print(f"Ground truth: {len(ground_truth)} chars")

    # Create run manifest
    manifest = RunContract.create_run_manifest(
        model_id, Path(f'models/{model_id}/config.yaml'),
        audio_path, text_path
    )

    # Transcribe
    print("Transcribing...")

    # Get timing context
    timer = PerformanceTimer()

    with timer.time_operation(f"{model_id}_transcribe") as timing_container:
        # Choose transcription function
        if model_id == 'whisper':
            text = transcribe_whisper(model_wrapper, audio, sr)
        elif model_id == 'faster_whisper':
            text = transcribe_faster_whisper(model_wrapper, audio, sr)
        elif model_id == 'seamlessm4t':
            text = transcribe_seamlessm4t(model_wrapper, audio, sr)
        elif model_id == 'lfm2_5_audio':
            text = transcribe_lfm2_5_audio(model_wrapper, audio, sr)
        else:
            raise ValueError(f"Unknown model: {model_id}")

    timing = timing_container['result']
    latency_ms = timing.elapsed_time_ms
    print(f"✓ Transcription: {len(text)} chars in {latency_ms:.1f}ms")

    # Apply normalization protocol
    if ground_truth:
        normalized_ref = NormalizationValidator.normalize_text(ground_truth)
        normalized_hyp = NormalizationValidator.normalize_text(text)
        print(f"✓ Normalization applied (protocol v{NormalizationValidator.NORMALIZATION_PROTOCOL['version']})")

    # Calculate metrics if ground truth available
    result = {
        'provider_id': model_id,
        'capability': 'asr',
        'input': {
            'audio_file': str(audio_path.name),
            'duration_s': metadata['duration_seconds'],
            'sr': sr
        },
        'output': {
            'text': text,
            'text_length': len(text),
            'normalized_text': normalized_hyp if ground_truth else None
        },
        'metrics': {
            'latency_ms_p50': latency_ms,
            'rtf': latency_ms / 1000 / metadata['duration_seconds']
        },
        'system': {
            'device': device,
            'model': config['model_name'],
            'inference_type': 'local'  # All current models are local
        },
        'protocol': {
            'normalization_version': NormalizationValidator.NORMALIZATION_PROTOCOL['version'],
            'entity_protocol_version': '1.0'  # From harness.protocol
        },
        'manifest': manifest,
        'timestamps': {
            'started_at': datetime.now().isoformat(),
            'finished_at': datetime.now().isoformat()
        },
        'errors': []
    }

    if ground_truth:
        # Use normalized text for metrics
        asr_result = ASRMetrics.evaluate(
            transcription=normalized_hyp,
            ground_truth=normalized_ref,
            audio_duration_s=metadata['duration_seconds'],
            latency_s=latency_ms / 1000
        )

        result['metrics']['wer'] = asr_result.wer
        result['metrics']['cer'] = asr_result.cer
        result['output']['ground_truth'] = ground_truth
        result['output']['ground_truth_normalized'] = normalized_ref
        result['output']['ground_truth_length'] = len(ground_truth)

        # Create validation report
        validation = create_validation_report(model_id, ground_truth, text)
        result['validation'] = validation

        print(f"WER: {asr_result.wer:.3f} ({asr_result.wer*100:.1f}%)")
        print(f"CER: {asr_result.cer:.3f} ({asr_result.cer*100:.1f}%)")
        print(f"RTF: {asr_result.rtv:.3f}x")

    # Save results
    results_dir = Path(f'runs/{model_id}/asr')
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_file = results_dir / f'{timestamp}.json'

    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"✓ Results saved to: {result_file}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Run ASR tests headless')
    parser.add_argument('--model', type=str, required=True,
                       choices=['lfm2_5_audio', 'whisper', 'faster_whisper', 'seamlessm4t'],
                       help='Model ID to test')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['smoke', 'primary', 'conversation'],
                       help='Dataset to test on')
    parser.add_argument('--device', type=str, default=None,
                       help='Override device (e.g., cpu, mps, cuda)')

    args = parser.parse_args()

    try:
        result = run_asr_test(args.model, args.dataset, device=args.device)
        print(f"\n🎉 Test completed successfully!")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())