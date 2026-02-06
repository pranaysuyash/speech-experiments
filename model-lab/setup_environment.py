#!/usr/bin/env python3
"""
Setup script for model testing lab environment.
Validates installation and creates necessary directories.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Check Python version compatibility."""
    print("=== Python Version Check ===")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major != 3 or version.minor < 9:
        print("❌ Requires Python 3.9 or higher")
        return False

    print("✓ Python version compatible")
    return True


def check_uv_installation():
    """Check if uv is installed."""
    print("\n=== UV Installation Check ===")
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ UV installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ UV not found")
            return False
    except FileNotFoundError:
        print("❌ UV not found")
        print("Install with: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False


def check_virtual_environment():
    """Check if virtual environment exists and is valid."""
    print("\n=== Virtual Environment Check ===")
    venv_path = Path(".venv")

    if not venv_path.exists():
        print("❌ Virtual environment not found")
        print("Create with: uv venv --python 3.11")
        return False

    # Check if Python exists in venv
    if sys.platform == "win32":
        python_path = venv_path / "Scripts" / "python.exe"
    else:
        python_path = venv_path / "bin" / "python"

    if not python_path.exists():
        print("❌ Virtual environment Python not found")
        return False

    print(f"✓ Virtual environment exists: {venv_path}")
    return True


def check_dependencies():
    """Check if required packages are installed."""
    print("\n=== Dependencies Check ===")

    required_packages = [
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "soundfile",
        "librosa",
        "requests",
        "pydantic",
        "jupyterlab",
        "psutil",
    ]

    missing_packages = []

    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)
        else:
            print(f"✓ {package}")

    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: uv pip install -r pyproject.toml")
        return False

    print("✓ All required packages installed")
    return True


def check_test_data():
    """Check if test data directory and files exist."""
    print("\n=== Test Data Check ===")

    data_dir = Path("data")
    audio_dir = data_dir / "audio"

    # Check directories
    if not data_dir.exists():
        print("❌ Data directory not found")
        return False

    if not audio_dir.exists():
        print("❌ Audio directory not found")
        print("Create with: python data/create_test_suite.py")
        return False

    # Check for test files
    required_files = [
        "test_manifest.json",
        "silence_5s.wav",
        "white_noise_10s.wav",
        "sine_sweep_20_2000hz_10s.wav",
    ]

    missing_files = []
    for file in required_files:
        if not (audio_dir / file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"❌ Missing test files: {', '.join(missing_files)}")
        print("Generate with: python data/create_test_suite.py")
        return False

    # Check for canonical audio (optional but recommended)
    canonical_audio = audio_dir / "clean_speech_10s.wav"
    canonical_text = data_dir / "text" / "clean_speech_10s.txt"

    if canonical_audio.exists() and canonical_text.exists():
        print("✓ Canonical test audio found")
    elif canonical_audio.exists():
        print("⚠️  Canonical audio found but missing text file")
        print("   Create clean_speech_10s.txt with exact transcription")
    else:
        print("⚠️  No canonical test audio found")
        print("   Record yourself reading Wikipedia text")
        print("   See data/RECORDING_INSTRUCTIONS.md")

    print(f"✓ Test data directory: {audio_dir}")
    print(f"✓ Found {len(list(audio_dir.glob('*.wav')))} audio files")
    return True


def check_harness_modules():
    """Check if harness modules are importable."""
    print("\n=== Harness Modules Check ===")

    harness_dir = Path("harness")
    if not harness_dir.exists():
        print("❌ Harness directory not found")
        return False

    required_modules = ["timers", "audio_io", "prompts", "evals"]
    missing_modules = []

    for module in required_modules:
        module_path = harness_dir / f"{module}.py"
        if not module_path.exists():
            missing_modules.append(module)
        else:
            print(f"✓ {module}.py")

    if missing_modules:
        print(f"❌ Missing modules: {', '.join(missing_modules)}")
        return False

    print("✓ All harness modules present")
    return True


def check_environment_variables():
    """Check for required environment variables."""
    print("\n=== Environment Variables Check ===")

    required_vars = ["LFM_AUDIO_API_KEY"]
    optional_vars = ["LFM_AUDIO_API_ENDPOINT"]

    missing_required = []

    for var in required_vars:
        if var not in os.environ:
            missing_required.append(var)
        else:
            print(f"✓ {var} (set)")

    for var in optional_vars:
        if var in os.environ:
            print(f"✓ {var} (set)")
        else:
            print(f"ℹ️  {var} (optional, not set)")

    if missing_required:
        print(f"❌ Missing required variables: {', '.join(missing_required)}")
        print("Set with: export LFM_AUDIO_API_KEY=your_key_here")
        return False

    print("✓ Environment variables configured")
    return True


def main():
    """Run all validation checks."""
    print("🔬 Model Testing Lab Environment Validation")
    print("=" * 50)

    checks = [
        check_python_version,
        check_uv_installation,
        check_virtual_environment,
        check_dependencies,
        check_test_data,
        check_harness_modules,
        check_environment_variables,
    ]

    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"❌ Check failed with error: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 All checks passed! ({passed}/{total})")
        print("\nNext steps:")
        print("1. Launch Jupyter: jupyter lab")
        print("2. Open notebooks/audio/lfm2_5_audio.ipynb")
        print("3. Follow the experiment workflow")
    else:
        print(f"⚠️  Some checks failed ({passed}/{total})")
        print("\nPlease fix the issues above before proceeding.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
