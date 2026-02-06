#!/usr/bin/env python3
"""
Create complete test audio suite for model evaluation.
Generates synthetic test files and provides framework for canonical recordings.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def generate_silence(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate complete silence."""
    return np.zeros(int(sample_rate * duration))


def generate_white_noise(
    duration: float, sample_rate: int = 16000, amplitude: float = 0.1
) -> np.ndarray:
    """Generate white noise."""
    return amplitude * np.random.randn(int(sample_rate * duration))


def generate_pink_noise(
    duration: float, sample_rate: int = 16000, amplitude: float = 0.1
) -> np.ndarray:
    """Generate pink noise (1/f noise)."""
    # Simple approximation using filtered white noise
    noise = np.random.randn(int(sample_rate * duration))
    # Basic pink noise filter
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]

    # Apply filter (simplified)
    pink_noise = np.convolve(noise, b, mode="same")
    return amplitude * pink_noise


def generate_sine_sweep(
    start_freq: float, end_freq: float, duration: float, sample_rate: int = 16000
) -> np.ndarray:
    """Generate logarithmic sine sweep."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Logarithmic frequency sweep
    k = (end_freq / start_freq) ** (1 / duration)
    instantaneous_freq = start_freq * (k**t)
    phase = 2 * np.pi * start_freq * (k**t - 1) / np.log(k)
    return 0.5 * np.sin(phase)


def generate_multitone(
    duration: float, sample_rate: int = 16000, frequencies: list[float] = None
) -> np.ndarray:
    """Generate multi-tone signal."""
    if frequencies is None:
        frequencies = [220, 440, 880, 1760]  # A3, A4, A5, A6

    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.zeros_like(t)

    for i, freq in enumerate(frequencies):
        amplitude = 1.0 / (i + 1)  # Decreasing amplitude
        signal += amplitude * np.sin(2 * np.pi * freq * t)

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    return signal


def generate_click_train(
    duration: float, sample_rate: int = 16000, click_rate: float = 10.0
) -> np.ndarray:
    """Generate periodic click train."""
    samples = int(sample_rate * duration)
    signal = np.zeros(samples)

    click_interval = int(sample_rate / click_rate)
    click_duration = int(0.001 * sample_rate)  # 1ms click

    for i in range(0, samples, click_interval):
        end = min(i + click_duration, samples)
        signal[i:end] = 1.0

    return signal * 0.5


def add_noise(clean_signal: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    """Add white noise to signal."""
    noise = np.random.randn(len(clean_signal))
    return clean_signal + noise_level * noise


def apply_fade(signal: np.ndarray, fade_duration: float, sample_rate: int) -> np.ndarray:
    """Apply fade in/out to signal."""
    fade_samples = int(fade_duration * sample_rate)

    if fade_samples > 0 and fade_samples < len(signal):
        # Fade in
        fade_in = np.linspace(0, 1, fade_samples)
        signal[:fade_samples] *= fade_in

        # Fade out
        fade_out = np.linspace(1, 0, fade_samples)
        signal[-fade_samples:] *= fade_out

    return signal


def create_test_suite(output_dir: Path, sample_rate: int = 16000) -> dict[str, Path]:
    """Create complete test audio suite."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_files = {}

    # Synthetic test signals
    synthetic_tests = [
        {
            "name": "silence_5s.wav",
            "generator": lambda: generate_silence(5.0, sample_rate),
            "description": "Complete silence for 5 seconds",
        },
        {
            "name": "white_noise_10s.wav",
            "generator": lambda: generate_white_noise(10.0, sample_rate, amplitude=0.2),
            "description": "White noise at 20% amplitude",
        },
        {
            "name": "pink_noise_10s.wav",
            "generator": lambda: generate_pink_noise(10.0, sample_rate, amplitude=0.2),
            "description": "Pink noise (1/f) at 20% amplitude",
        },
        {
            "name": "sine_sweep_20_2000hz_10s.wav",
            "generator": lambda: generate_sine_sweep(20, 2000, 10.0, sample_rate),
            "description": "Logarithmic sine sweep from 20Hz to 2kHz",
        },
        {
            "name": "multitone_10s.wav",
            "generator": lambda: generate_multitone(10.0, sample_rate),
            "description": "Multi-tone signal (A3, A4, A5, A6)",
        },
        {
            "name": "clicks_10s_10hz.wav",
            "generator": lambda: generate_click_train(10.0, sample_rate, click_rate=10.0),
            "description": "Click train at 10 Hz",
        },
    ]

    # Generate synthetic files
    for test_config in synthetic_tests:
        try:
            audio = test_config["generator"]()

            # Apply fade to avoid clicks
            audio = apply_fade(audio, 0.05, sample_rate)

            filepath = output_dir / test_config["name"]
            sf.write(filepath, audio, sample_rate)

            test_files[test_config["name"]] = {
                "path": filepath,
                "description": test_config["description"],
                "duration": len(audio) / sample_rate,
                "type": "synthetic",
            }

            print(f"✓ Generated: {test_config['name']}")

        except Exception as e:
            print(f"✗ Failed to generate {test_config['name']}: {e}")

    # Create noisy versions if clean speech exists
    clean_speech = output_dir / "clean_speech_10s.wav"
    if clean_speech.exists():
        try:
            clean_audio, sr = sf.read(clean_speech)
            if sr != sample_rate:
                # Resample if needed
                import librosa

                clean_audio = librosa.resample(clean_audio, orig_sr=sr, target_sr=sample_rate)

            # Generate noisy versions
            noise_levels = [0.1, 0.2, 0.5]
            for noise_level in noise_levels:
                noisy_audio = add_noise(clean_audio, noise_level)
                filename = f"noisy_speech_10s_{int(noise_level * 100)}pct.wav"
                filepath = output_dir / filename
                sf.write(filepath, noisy_audio, sample_rate)

                test_files[filename] = {
                    "path": filepath,
                    "description": f"Speech with {noise_level * 100:.0f}% white noise",
                    "duration": len(noisy_audio) / sample_rate,
                    "type": "noisy_speech",
                    "noise_level": noise_level,
                }
                print(f"✓ Generated: {filename}")

        except Exception as e:
            print(f"✗ Failed to process clean speech: {e}")

    return test_files


def create_manifest(test_files: dict[str, dict], output_path: Path) -> None:
    """Create test suite manifest file."""
    manifest = {
        "created_at": np.datetime64("now").astype(str),
        "sample_rate": 16000,
        "files": test_files,
    }

    import json

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="Create test audio suite")
    parser.add_argument(
        "--output-dir", type=str, default="audio", help="Output directory for test files"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Sample rate for generated files"
    )
    parser.add_argument("--list-only", action="store_true", help="List what would be generated")

    args = parser.parse_args()

    if args.list_only:
        print("Would generate the following test audio files:")
        test_configs = [
            ("silence_5s.wav", "Complete silence for 5 seconds"),
            ("white_noise_10s.wav", "White noise at 20% amplitude"),
            ("pink_noise_10s.wav", "Pink noise (1/f) at 20% amplitude"),
            ("sine_sweep_20_2000hz_10s.wav", "Logarithmic sine sweep from 20Hz to 2kHz"),
            ("multitone_10s.wav", "Multi-tone signal (A3, A4, A5, A6)"),
            ("clicks_10s_10hz.wav", "Click train at 10 Hz"),
        ]
        for name, desc in test_configs:
            print(f"  - {name}: {desc}")
        print("\nIf clean_speech_10s.wav exists, would also generate:")
        print("  - noisy_speech_10s_10pct.wav: Speech with 10% white noise")
        print("  - noisy_speech_10s_20pct.wav: Speech with 20% white noise")
        print("  - noisy_speech_10s_50pct.wav: Speech with 50% white noise")
    else:
        output_dir = Path(args.output_dir)
        print(f"Creating test audio suite in {output_dir}...")

        test_files = create_test_suite(output_dir, args.sample_rate)

        # Create manifest
        manifest_path = output_dir / "test_manifest.json"
        create_manifest(test_files, manifest_path)

        print(f"\n✓ Created {len(test_files)} test audio files")
        print(f"✓ Manifest saved to {manifest_path}")

        # Print summary
        print("\nGenerated files:")
        for filename, info in test_files.items():
            print(f"  - {filename}: {info['description']}")

        if "clean_speech_10s.wav" not in test_files:
            print("\n⚠️  No clean_speech_10s.wav found. Please record your voice reading")
            print("   Wikipedia text and save as clean_speech_10s.wav")
            print("   See RECORDING_INSTRUCTIONS.md for details")


if __name__ == "__main__":
    main()
