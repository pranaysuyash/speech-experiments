#!/usr/bin/env python3
"""
Generate test audio files for model testing.
Creates synthetic audio with known characteristics for systematic evaluation.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import argparse


def generate_sine_wave(frequency: float, duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate a sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    return np.sin(2 * np.pi * frequency * t)


def generate_chirp(start_freq: float, end_freq: float, duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate a linear chirp."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    return np.sin(2 * np.pi * (start_freq + (end_freq - start_freq) * t / (2 * duration)) * t)


def generate_white_noise(duration: float, sample_rate: int = 16000, amplitude: float = 0.1) -> np.ndarray:
    """Generate white noise."""
    return amplitude * np.random.randn(int(sample_rate * duration))


def generate_speech_like(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate speech-like signal with multiple frequency components."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Fundamental frequency (typical male speaker)
    f0 = 120
    
    # Formant frequencies
    f1, f2, f3 = 500, 1500, 2500
    
    # Generate harmonics
    signal = (0.5 * np.sin(2 * np.pi * f0 * t) +
              0.3 * np.sin(2 * np.pi * 2 * f0 * t) +
              0.2 * np.sin(2 * np.pi * 3 * f0 * t))
    
    # Add formant structure
    formants = (0.4 * np.sin(2 * np.pi * f1 * t) +
                0.3 * np.sin(2 * np.pi * f2 * t) +
                0.2 * np.sin(2 * np.pi * f3 * t))
    
    # Combine with envelope
    envelope = np.abs(np.sin(2 * np.pi * 3 * t))  # 3 Hz modulation
    
    return 0.7 * signal + 0.3 * formants * envelope


def generate_test_audio_files(output_dir: Path):
    """Generate a suite of test audio files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sample_rate = 16000
    
    # Test files with different characteristics
    test_files = [
        {
            'name': 'sine_440hz_5s.wav',
            'generator': lambda: generate_sine_wave(440, 5.0, sample_rate),
            'description': 'Pure 440Hz sine wave (A4 note)'
        },
        {
            'name': 'chirp_200_2000hz_3s.wav',
            'generator': lambda: generate_chirp(200, 2000, 3.0, sample_rate),
            'description': 'Linear chirp from 200Hz to 2000Hz'
        },
        {
            'name': 'white_noise_2s.wav',
            'generator': lambda: generate_white_noise(2.0, sample_rate, amplitude=0.2),
            'description': 'White noise at 20% amplitude'
        },
        {
            'name': 'speech_like_4s.wav',
            'generator': lambda: generate_speech_like(4.0, sample_rate),
            'description': 'Synthetic speech-like signal'
        },
        {
            'name': 'multi_tone_6s.wav',
            'generator': lambda: (generate_sine_wave(261.63, 2.0, sample_rate) * 0.4 +  # C4
                                generate_sine_wave(329.63, 2.0, sample_rate) * 0.3 +  # E4
                                generate_sine_wave(392.00, 2.0, sample_rate) * 0.3), # G4
            'description': 'C major chord (C-E-G)'
        }
    ]
    
    generated_files = []
    
    for test_file in test_files:
        try:
            # Generate audio
            audio = test_file['generator']()
            
            # Ensure proper shape (samples, channels)
            if audio.ndim == 1:
                audio = audio.reshape(-1, 1)
            
            # Save to file
            file_path = output_dir / test_file['name']
            sf.write(file_path, audio, sample_rate)
            
            generated_files.append({
                'file': file_path,
                'description': test_file['description'],
                'duration': len(audio) / sample_rate,
                'sample_rate': sample_rate
            })
            
            print(f"✓ Generated: {test_file['name']} ({test_file['description']})")
            
        except Exception as e:
            print(f"✗ Failed to generate {test_file['name']}: {e}")
    
    return generated_files


def main():
    parser = argparse.ArgumentParser(description="Generate test audio files")
    parser.add_argument('--output-dir', type=str, default='../data/audio',
                       help='Output directory for audio files')
    parser.add_argument('--list-only', action='store_true',
                       help='List what would be generated without creating files')
    
    args = parser.parse_args()
    
    if args.list_only:
        print("Would generate the following test audio files:")
        test_files = [
            ('sine_440hz_5s.wav', 'Pure 440Hz sine wave (A4 note)'),
            ('chirp_200_2000hz_3s.wav', 'Linear chirp from 200Hz to 2000Hz'),
            ('white_noise_2s.wav', 'White noise at 20% amplitude'),
            ('speech_like_4s.wav', 'Synthetic speech-like signal'),
            ('multi_tone_6s.wav', 'C major chord (C-E-G)')
        ]
        for name, desc in test_files:
            print(f"  - {name}: {desc}")
    else:
        print(f"Generating test audio files in {args.output_dir}...")
        generated = generate_test_audio_files(args.output_dir)
        print(f"\\n✓ Generated {len(generated)} test audio files")
        
        # Print summary
        print("\\nGenerated files:")
        for file_info in generated:
            print(f"  - {file_info['file'].name}: {file_info['description']}")


if __name__ == '__main__':
    main()