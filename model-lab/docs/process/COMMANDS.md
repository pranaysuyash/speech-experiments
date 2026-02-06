# Command Toolkit (rg-first)

This repo prefers fast, reproducible discovery commands. Default to `rg` over `grep`.

## Always Run (First 2 minutes)

```bash
# Ticket scan (helps avoid collisions and see current work)
rg -n "TCK-\\d{8}-\\d{3}" docs/WORKLOG_TICKETS.md | tail -n 30

# Find TODOs / fixmes
rg -n "TODO|FIXME|HACK" -S server harness client tests || true

# Quick inventory (docs + prompts)
find docs -maxdepth 2 -type f -name '*.md' | sort
find prompts -maxdepth 3 -type f -name '*.md' | sort
```

## Local Workflow Gate (No PR Required)

```bash
# Ensure repo-managed hooks are enabled
git config core.hooksPath .githooks
git config --get core.hooksPath

# Run the workflow gate against staged changes
./scripts/agent_gate.sh --staged
```

## Canonical File Finding

```bash
# Find where a feature is actually wired/used
rg -n "<keyword>" -S server harness

# API endpoints
rg -n "@router\\.(get|post|put|delete)|@app\\.(get|post)" -S server/api

# Model registry entries
rg -n "register_model|MODEL_REGISTRY" -S harness models
```

## "No Parallel Versions" Check

```bash
# Detect suspicious duplicates
find . -maxdepth 8 -type f \( -name '*_v2.*' -o -name '*_new.*' -o -name '*copy*' -o -name '*backup*' -o -name '*old*' \) | grep -v node_modules | grep -v .venv
```

## Backend Verification (preferred)

```bash
# Core invariants
PYTHONPATH=. pytest -q tests/integration/test_backend_invariants.py

# Security tests
PYTHONPATH=. pytest -q tests/api/test_artifact_download_security.py

# Full test suite
PYTHONPATH=. pytest -q tests/

# Type check
PYTHONPATH=. mypy server/ harness/ --ignore-missing-imports
```

## Frontend Verification (preferred)

```bash
cd client && npm run build
cd client && npm run lint
cd client && npm run type-check
```

## Model Inference Verification

```bash
# === AUDIO ===
# Speech recognition
PYTHONPATH=. python -m harness.run --model whisper-large-v3 --input inputs/test.wav

# Music generation
PYTHONPATH=. python -m harness.run --model musicgen-large --prompt "upbeat jazz" --duration 10

# Audio separation
PYTHONPATH=. python -m harness.run --model demucs --input inputs/song.mp3 --task separate

# Voice cloning
PYTHONPATH=. python -m harness.run --model xtts --input inputs/voice.wav --text "Hello world"

# === VISION ===
# Image classification
PYTHONPATH=. python -m harness.run --model clip-vit-large --input inputs/test.jpg

# Object detection
PYTHONPATH=. python -m harness.run --model yolov8 --input inputs/test.jpg

# Image generation
PYTHONPATH=. python -m harness.run --model sdxl --prompt "a cat" 

# Depth estimation
PYTHONPATH=. python -m harness.run --model depth-anything --input inputs/test.jpg

# Pose estimation
PYTHONPATH=. python -m harness.run --model mediapipe-pose --input inputs/person.jpg

# Super-resolution
PYTHONPATH=. python -m harness.run --model real-esrgan --input inputs/low-res.jpg

# === VIDEO ===
# Action recognition
PYTHONPATH=. python -m harness.run --model video-mae --input inputs/video.mp4

# Object tracking
PYTHONPATH=. python -m harness.run --model sam2 --input inputs/video.mp4

# === MULTIMODAL ===
# Image captioning
PYTHONPATH=. python -m harness.run --model blip2 --input inputs/test.jpg --task caption

# Visual QA
PYTHONPATH=. python -m harness.run --model llava --input inputs/test.jpg --prompt "What's in this image?"

# === GENERATIVE ===
# 3D generation
PYTHONPATH=. python -m harness.run --model point-e --prompt "a chair"

# Code generation
PYTHONPATH=. python -m harness.run --model codellama --prompt "def fibonacci(n):"

# === EMBEDDINGS ===
# Get embeddings
PYTHONPATH=. python -m harness.run --model sentence-transformers --input "Hello world" --task embed

# === UTILITIES ===
# List all models
PYTHONPATH=. python -m harness.list_models

# List by domain
PYTHONPATH=. python -m harness.list_models --domain audio

# Model info
PYTHONPATH=. python -m harness.model_info --model <model-name>

# Compare models
PYTHONPATH=. python -m compare.run --models model1,model2 --input inputs/test.wav
```

## Notes

- If `rg` is missing, install ripgrep or document the blocker as `Unknown`.
- If git is unavailable, record it and avoid git-only claims.
- Always verify model inference on the target hardware before claiming performance metrics.
