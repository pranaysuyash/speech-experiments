#!/usr/bin/env python3
"""
Wrapper for zai-org/GLM-TTSRepo Runner.
Adapts standard harness inputs to GLM-TTS repo pipeline.
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
import tempfile
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("glm_tts_wrapper")


def check_weights(ckpt_dir):
    """Verify weights exist."""
    required_files = [
        "llm/model.safetensors.index.json",
        "flow/flow.pt",
        "speech_tokenizer/model.safetensors",
    ]

    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        return False, f"Checkpoint directory not found: {ckpt_dir}"

    # Check for non-empty directory at least
    if not any(ckpt_path.iterdir()):
        return False, f"Checkpoint directory is empty: {ckpt_dir}"

    return True, "OK"


def generate_jsonl(text, output_dir, prompt_wav=None, prompt_text=None):
    """Generate the input JSONL for glmtts_inference.py"""

    # Use default prompt if none provided
    # The repo provides examples/prompt/jiayan_zh.wav
    repo_root = Path(__file__).parent / "repo"
    default_prompt_wav = repo_root / "examples" / "prompt" / "jiayan_zh.wav"
    default_prompt_text = "燃放爆竹。燃放爆竹。"  # From their example

    if not prompt_wav:
        prompt_wav = str(default_prompt_wav)
    if not prompt_text:
        prompt_text = default_prompt_text

    entry = {
        "uttid": "test_001",
        "syn_text": text,
        "prompt_speech": prompt_wav,
        "prompt_text": prompt_text,
    }

    jsonl_path = Path(output_dir) / "input.jsonl"
    with open(jsonl_path, "w") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return str(jsonl_path)


def get_glm_venv_python():
    """Get the Python executable from the GLM-TTS specific venv."""
    wrapper_dir = Path(__file__).parent
    venv_python = wrapper_dir / "venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    # Fallback to system python if venv doesn't exist
    return sys.executable


def run_inference(jsonl_path, output_dir, ckpt_dir):
    """Run official inference script."""
    repo_root = Path(__file__).parent / "repo"
    inference_script = repo_root / "glmtts_inference.py"

    # Use GLM-TTS specific venv
    glm_python = get_glm_venv_python()

    # Creating a temp 'examples' dir link because script seemingly expects it relative?
    # Actually script takes --data <name-of-jsonl-file-without-extension>
    # and expects it in 'examples/<name>.jsonl'
    # This is rigid. Let's symlink our jsonl into repo/examples/temp_bench.jsonl

    data_name = "temp_bench"
    data_link = repo_root / "examples" / f"{data_name}.jsonl"

    # Cleanup previous link (handle broken symlinks too)
    data_link.unlink(missing_ok=True)

    # Symlink our input
    # Note: symlink target must be absolute or relative to link location
    os.symlink(os.path.abspath(jsonl_path), data_link)

    # The script output dir logic:
    # outputs/pretrain{exp_name}/{data_name}
    # We can control exp_name.
    exp_name = "_bench"

    cmd = [
        glm_python,
        str(inference_script),
        "--data",
        data_name,
        "--exp_name",
        exp_name,
        "--use_cache",
        # ckpt dir is hardcoded in the script to be 'ckpt/' relative to script?
        # script says: os.path.join('ckpt', ...)
    ]

    # We must ensure 'ckpt' exists in repo root.
    # We can symlink our external ckpt dir to repo/ckpt
    repo_ckpt_link = repo_root / "ckpt"

    if repo_ckpt_link.is_symlink():
        repo_ckpt_link.unlink()
    elif repo_ckpt_link.exists() and not repo_ckpt_link.is_symlink():
        # If it's a real dir, unexpected but leave it if not empty?
        # But we want to enforce OUR weights.
        # Warn?
        logger.warning(f"Repo has existing ckpt dir: {repo_ckpt_link}")
        # We should probably use it if verify passes, or fail.

    if not repo_ckpt_link.exists():
        os.symlink(os.path.abspath(ckpt_dir), repo_ckpt_link)

    logger.info(f"Running inference: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"GLM-TTS inference failed: {result.stderr}")

    # Locate output
    # repo_root / outputs / pretrain_bench / temp_bench / test_001.wav
    expected_wav = repo_root / "outputs" / f"pretrain{exp_name}" / data_name / "test_001.wav"

    if not expected_wav.exists():
        raise RuntimeError(
            f"Output WAV not found at {expected_wav}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    return str(expected_wav)


def validate_audio(wav_path):
    """Check if audio is valid using soundfile."""
    import soundfile as sf
    import numpy as np

    try:
        data, sr = sf.read(wav_path)
        duration = len(data) / sr

        if duration < 0.2:
            return False, f"Duration too short: {duration:.2f}s"

        if np.max(np.abs(data)) == 0:
            return False, "Audio is silent"

        return True, {
            "duration": duration,
            "sample_rate": sr,
            "channels": data.shape[1] if len(data.shape) > 1 else 1,
        }
    except Exception as e:
        return False, f"Audio validation failed: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ckpt_dir", default=None)  # Optional override
    args = parser.parse_args()

    # Default ckpt dir to <script_dir>/ckpt if not provided
    if not args.ckpt_dir:
        args.ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt")

    # 1. Check Weights
    ok, msg = check_weights(args.ckpt_dir)
    if not ok:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "error": f"Weights missing. Run: huggingface-cli download zai-org/GLM-TTS --local-dir {args.ckpt_dir}",
                }
            )
        )
        return

    try:
        with tempfile.TemporaryDirectory() as temp_input_dir:
            # 2. Generate Input
            jsonl_path = generate_jsonl(args.text, temp_input_dir)

            # 3. Run Inference
            wav_path = run_inference(jsonl_path, args.output_dir, args.ckpt_dir)

            # 4. Validate output
            valid, info = validate_audio(wav_path)
            if not valid:
                print(json.dumps({"status": "error", "error": str(info)}))
                return

            # 5. Move to requested output (or read data)
            # The harness might expect raw audio data or a file path.
            # Bench runner "proxy" might expect audio array?
            # But the runner interface for invoke_repo_wrapper returns JSON.
            # Best to return path + read data here?

            import soundfile as sf

            data, sr = sf.read(wav_path)

            # Return result
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "audio_path": wav_path,  # Path in repo output dir
                        "sample_rate": sr,
                        "duration": info["duration"],
                    }
                )
            )

    except Exception as e:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": str(e),  # + f"\nTraceback: {traceback.format_exc()}"
                }
            )
        )


if __name__ == "__main__":
    main()
