#!/usr/bin/env python3
"""
Create smoke test dataset from primary dataset.
Uses first 10 seconds of primary recording for quick validation.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import hashlib

def create_smoke_dataset():
    """Create smoke test dataset from available audio."""

    print("=== Creating Smoke Test Dataset ===")

    # Use available WAV file for smoke test
    smoke_audio_source = Path('data/audio/conversation_2ppl_30s.wav')
    smoke_text_source = Path('data/text/GROUND_TRUTH/conversation_2ppl_30s.txt')

    if not smoke_audio_source.exists():
        print(f"✗ Smoke source audio not found: {smoke_audio_source}")
        print("ℹ️  Available audio files:")
        import os
        audio_dir = Path('data/audio')
        for wav_file in audio_dir.glob('*.wav'):
            print(f"   - {wav_file}")
        return False

    # Load audio
    try:
        audio, sr = sf.read(smoke_audio_source)
        print(f"✓ Loaded source audio: {len(audio)/sr:.1f}s @ {sr}Hz")
    except Exception as e:
        print(f"✗ Failed to load audio: {e}")
        return False

    # Extract first 10 seconds for smoke test
    smoke_audio = audio[:10 * sr]
    print(f"✓ Extracted 10s smoke test")

    # Create smoke directory
    smoke_dir = Path('data/audio/SMOKE')
    smoke_dir.mkdir(parents=True, exist_ok=True)

    # Save smoke audio
    smoke_audio_path = smoke_dir / 'conversation_2ppl_10s.wav'
    sf.write(smoke_audio_path, smoke_audio, sr)
    print(f"✓ Saved smoke audio: {smoke_audio_path}")

    # Create smoke text (first ~200 chars of ground truth if available)
    smoke_text_path = None
    smoke_text = "This is a smoke test for automatic speech recognition validation. The quick brown fox jumps over the lazy dog. Testing entity extraction: numbers like 123 and 45.67, dates like 01/08/2024, and currency like $19.99."

    if smoke_text_source.exists():
        try:
            with open(smoke_text_source, 'r') as f:
                full_text = f.read().strip()

            # Create 10s equivalent text (roughly 200 chars for normal speech)
            smoke_text = full_text[:200]
            if len(smoke_text) < len(full_text):
                # Try to end at a sentence boundary
                last_period = smoke_text.rfind('.')
                if last_period > 100:  # Only if we get a reasonable chunk
                    smoke_text = smoke_text[:last_period + 1]

            print(f"✓ Using ground truth text source")
        except Exception as e:
            print(f"⚠️  Could not load text source: {e}")
            print(f"   Using default smoke text with entities")

    # Save smoke text
    smoke_text_dir = Path('data/text/SMOKE')
    smoke_text_dir.mkdir(parents=True, exist_ok=True)

    smoke_text_path = smoke_text_dir / 'conversation_2ppl_10s.txt'
    with open(smoke_text_path, 'w') as f:
        f.write(smoke_text.strip())

    print(f"✓ Saved smoke text: {smoke_text_path} ({len(smoke_text)} chars)")

    # Calculate dataset hash
    dataset_content = f"{smoke_audio_path.name}_{smoke_text}"
    dataset_hash = hashlib.sha256(dataset_content.encode()).hexdigest()[:16]
    print(f"✓ Dataset hash: {dataset_hash}")

    print("\n🎉 Smoke test dataset created successfully!")
    print(f"   Audio: {smoke_audio_path}")
    print(f"   Text: {smoke_text_path}")
    print(f"   Duration: {len(smoke_audio)/sr:.1f}s")
    print(f"   Hash: {dataset_hash}")

    return True

if __name__ == '__main__':
    import sys
    sys.exit(0 if create_smoke_dataset() else 1)