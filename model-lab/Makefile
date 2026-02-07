# Model Lab Makefile
# Single entrypoint for all operations

.PHONY: setup setup-kernel lint format test lab asr tts scorecard list-models clean

# Default Python for uv
PYTHON := uv run python

# =============================================================================
# Setup
# =============================================================================

setup:
	uv sync
	@echo "✓ Environment synced"

setup-kernel:
	$(PYTHON) -m ipykernel install --user --name=model-lab --display-name="Model Lab (Python 3.12)"
	@echo "✓ Jupyter kernel installed"

# =============================================================================
# Code Quality
# =============================================================================

lint:
	uv run ruff check .
	@echo "✓ Lint passed"

format:
	uv run ruff format .
	uv run ruff check . --fix
	@echo "✓ Formatted"

# =============================================================================
# Testing
# =============================================================================

test:
	$(PYTHON) -m pytest -q tests/
	@echo "✓ Tests passed"

test-imports:
	$(PYTHON) -c "from harness import registry, contracts, metrics_asr, metrics_tts; print('✓ All imports OK')"

# =============================================================================
# Development
# =============================================================================

lab:
	uv run jupyter lab

notebook:
	uv run jupyter notebook

# =============================================================================
# Model Operations
# =============================================================================

# Usage: make asr MODEL=whisper DATASET=llm_primary
asr:
ifndef MODEL
	$(error MODEL is required. Usage: make asr MODEL=whisper DATASET=llm_primary)
endif
ifndef DATASET
	$(error DATASET is required. Usage: make asr MODEL=whisper DATASET=llm_primary)
endif
	$(PYTHON) scripts/run_asr.py --model $(MODEL) --dataset $(DATASET)

# Usage: make asr-audio MODEL=moonshine AUDIO=inputs/sample_16k.wav
asr-audio:
ifndef MODEL
	$(error MODEL is required. Usage: make asr-audio MODEL=moonshine AUDIO=inputs/sample.wav)
endif
ifndef AUDIO
	$(error AUDIO is required. Usage: make asr-audio MODEL=moonshine AUDIO=inputs/sample.wav)
endif
	$(PYTHON) -c "\
import soundfile as sf; \
from harness.registry import ModelRegistry; \
audio, sr = sf.read('$(AUDIO)'); \
bundle = ModelRegistry.load_model('$(MODEL)', {}, device='cpu'); \
result = bundle['asr']['transcribe'](audio, sr=sr); \
print('Text:', result.get('text', result))"

# Usage: make asr-stream-audio MODEL=kyutai_streaming AUDIO=inputs/sample_16k.wav [CHUNK_MS=160]
asr-stream-audio:
ifndef MODEL
	$(error MODEL is required. Usage: make asr-stream-audio MODEL=kyutai_streaming AUDIO=inputs/sample.wav)
endif
ifndef AUDIO
	$(error AUDIO is required. Usage: make asr-stream-audio MODEL=kyutai_streaming AUDIO=inputs/sample.wav)
endif
	$(PYTHON) -c "\
import json; \
import soundfile as sf; \
from harness.registry import ModelRegistry; \
from harness.streaming_metrics import measure_streaming_latency; \
audio, sr = sf.read('$(AUDIO)'); \
bundle = ModelRegistry.load_model('$(MODEL)', {}, device='cpu'); \
metrics = measure_streaming_latency(bundle['asr_stream'], audio, sr, chunk_ms=$(if $(CHUNK_MS),$(CHUNK_MS),160)); \
print(json.dumps(metrics, indent=2))"

# =============================================================================
# Benchmarks (LCS-B)
# =============================================================================

# Usage: make bench-asr-stream MODEL=kyutai_streaming AUDIO=data/audio/clean_speech_10s.wav [REF="reference text"]
bench-asr-stream:
ifndef MODEL
	$(error MODEL is required. Usage: make bench-asr-stream MODEL=kyutai_streaming AUDIO=audio.wav)
endif
ifndef AUDIO
	$(error AUDIO is required. Usage: make bench-asr-stream MODEL=kyutai_streaming AUDIO=audio.wav)
endif
	$(PYTHON) -c "\
import json; \
from bench.runner import run_streaming_asr_bench, save_result; \
result = run_streaming_asr_bench('$(MODEL)', '$(AUDIO)', reference='$(REF)' if '$(REF)' else None); \
save_result(result); \
print(json.dumps(result, indent=2))"

# Usage: make bench-asr MODEL=faster_whisper AUDIO=data/audio/clean_speech_10s.wav [REF="reference text"]
bench-asr:
ifndef MODEL
	$(error MODEL is required. Usage: make bench-asr MODEL=faster_whisper AUDIO=audio.wav)
endif
ifndef AUDIO
	$(error AUDIO is required. Usage: make bench-asr MODEL=faster_whisper AUDIO=audio.wav)
endif
	$(PYTHON) -c "\
import json; \
from bench.runner import run_batch_asr_bench, save_result; \
result = run_batch_asr_bench('$(MODEL)', '$(AUDIO)', reference='$(REF)' if '$(REF)' else None); \
save_result(result); \
print(json.dumps(result, indent=2))"

# Usage: make tts MODEL=lfm2_5_audio TEXT="Hello world"
tts:
ifndef MODEL
	$(error MODEL is required. Usage: make tts MODEL=lfm2_5_audio TEXT="Hello")
endif
	$(PYTHON) scripts/run_tts.py --model $(MODEL) --text "$(TEXT)"

# =============================================================================
# Analysis
# =============================================================================

scorecard:
	$(PYTHON) -c "import subprocess; subprocess.run(['uv', 'run', 'jupyter', 'nbconvert', '--execute', '--to', 'html', 'compare/00_scorecard.ipynb'])"
	@echo "✓ Scorecard generated"

arsenal:
	$(PYTHON) scripts/generate_arsenal.py
	@echo "✓ Arsenal docs regenerated"

list-models:
	$(PYTHON) -c "from harness.registry import ModelRegistry; print('Registered models:', ModelRegistry.list_models())"

model-info:
ifndef MODEL
	$(error MODEL is required. Usage: make model-info MODEL=whisper)
endif
	$(PYTHON) -c "from harness.registry import ModelRegistry; import json; print(json.dumps(ModelRegistry.get_model_metadata('$(MODEL)'), indent=2))"

# Usage: make run-pipeline PIPELINE=config/pipelines/enhance_asr.yaml AUDIO=inputs/sample.wav
run-pipeline:
ifndef PIPELINE
	$(error PIPELINE is required. Usage: make run-pipeline PIPELINE=config/pipelines/enhance_asr.yaml AUDIO=inputs/sample.wav)
endif
ifndef AUDIO
	$(error AUDIO is required. Usage: make run-pipeline PIPELINE=config/pipelines/enhance_asr.yaml AUDIO=inputs/sample.wav)
endif
	$(PYTHON) -c "\
import soundfile as sf; \
import numpy as np; \
from harness.pipeline import run_pipeline; \
audio, sr = sf.read('$(AUDIO)'); \
result = run_pipeline('$(PIPELINE)', audio, sr); \
print('Steps:', result.steps_executed); \
print('Final:', result.final)"

# Usage: make model-install MODEL=demucs
model-install:
ifndef MODEL
	$(error MODEL is required. Usage: make model-install MODEL=demucs)
endif
	pip install -r models/$(MODEL)/requirements.txt
	@echo "✓ Model $(MODEL) dependencies installed"

# =============================================================================
# Cleanup
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned"

clean-runs:
	@echo "⚠️  This will delete all run artifacts!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] && rm -rf runs/*/* || echo "Cancelled"

# =============================================================================
# Help
# =============================================================================

help:
	@echo "Model Lab Commands:"
	@echo ""
	@echo "  Setup:"
	@echo "    make setup         - Sync environment with uv"
	@echo "    make setup-kernel  - Install Jupyter kernel"
	@echo ""
	@echo "  Development:"
	@echo "    make lint          - Check code with ruff"
	@echo "    make format        - Format code with ruff"
	@echo "    make test          - Run pytest"
	@echo "    make lab           - Start Jupyter Lab"
	@echo ""
	@echo "  Model Operations:"
	@echo "    make asr MODEL=whisper DATASET=llm_primary"
	@echo "    make tts MODEL=lfm2_5_audio TEXT='Hello'"
	@echo "    make list-models   - List registered models"
	@echo "    make model-info MODEL=whisper"
	@echo "    make model-install MODEL=demucs"
	@echo ""
	@echo "  Pipeline:"
	@echo "    make run-pipeline PIPELINE=config/pipelines/enhance_asr.yaml AUDIO=inputs/sample.wav"
	@echo ""
	@echo "  Analysis:"
	@echo "    make scorecard     - Generate comparison scorecard"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean         - Remove cache files"
