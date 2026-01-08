#!/usr/bin/env python3
"""
Convert m4a audio files to WAV format for model testing.
Uses ffmpeg for conversion if available.
"""

import subprocess
from pathlib import Path
import sys

def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        subprocess.run(['ffmpeg', '-version'],
                       capture_output=True,
                       check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def convert_m4a_to_wav(input_file: Path, output_file: Path):
    """Convert m4a file to WAV using ffmpeg."""
    if not input_file.exists():
        print(f"✗ Input file not found: {input_file}")
        return False

    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert using ffmpeg
    try:
        cmd = [
            'ffmpeg',
            '-i', str(input_file),
            '-ar', '16000',  # 16kHz sample rate (Whisper standard)
            '-ac', '1',       # Mono audio
            '-y',            # Overwrite output
            str(output_file)
        ]

        print(f"Converting: {input_file.name} → {output_file.name}")
        subprocess.run(cmd, capture_output=True, check=True)
        print(f"✓ Converted: {output_file}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Conversion failed: {e}")
        return False

def convert_primary_dataset():
    """Convert primary dataset m4a files to WAV."""
    print("=== Converting Primary Dataset ===")

    # Check ffmpeg availability
    if not check_ffmpeg():
        print("⚠️  ffmpeg not found. Install with: brew install ffmpeg")
        print("   Will use available WAV files instead.")
        return False

    conversions = [
        {
            'input': Path('data/audio/PRIMARY/llm_recording_pranay.m4a'),
            'output': Path('data/audio/PRIMARY/llm_recording_pranay.wav')
        },
        {
            'input': Path('data/audio/PRIMARY/UX_Psychology_From_Miller_s_Law_to_AI.m4a'),
            'output': Path('data/audio/PRIMARY/UX_Psychology_From_Miller_s_Law_to_AI.wav')
        }
    ]

    success_count = 0
    for conversion in conversions:
        if conversion['input'].exists():
            if convert_m4a_to_wav(conversion['input'], conversion['output']):
                success_count += 1
        else:
            print(f"⚠️  File not found: {conversion['input']}")

    print(f"\n✓ Converted {success_count}/{len(conversions)} files")

    return success_count > 0

if __name__ == '__main__':
    success = convert_primary_dataset()
    sys.exit(0 if success else 1)