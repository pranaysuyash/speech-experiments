# Model Lab Makefile
# Single entrypoint for all operations

.PHONY: setup setup-kernel lint format test lab asr tts scorecard list-models clean \
	hf-preflight hf-plan hf-run-all hf-report hf-prefetch

# Default Python for uv
PYTHON := uv run python
RUNNER := $(PYTHON) scripts/run_bench.py

# =============================================================================
# Setup
# =============================================================================

setup:
	uv sync --extra dev --group dev
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

# Usage: make bench-asr MODEL=faster_whisper_large_v3 AUDIO=data/audio/clean_speech_10s.wav [REF="reference text"]
bench-asr:
ifndef MODEL
	$(error MODEL is required. Usage: make bench-asr MODEL=faster_whisper_large_v3 AUDIO=audio.wav)
endif
ifndef AUDIO
	$(error AUDIO is required. Usage: make bench-asr MODEL=faster_whisper_large_v3 AUDIO=audio.wav)
endif
	$(PYTHON) -c "\
import json; \
from bench.runner import run_batch_asr_bench, save_result; \
result = run_batch_asr_bench('$(MODEL)', '$(AUDIO)', reference='$(REF)' if '$(REF)' else None, device='$(if $(DEVICE),$(DEVICE),cpu)'); \
save_result(result); \
print(json.dumps(result, indent=2))"

# Usage: make bench-asr-long MODEL=faster_whisper_large_v3 [AUDIO=data/audio/long_audio.wav] [CHUNK=30] [STRIDE=5] [BATCH=16] [DEVICE=cpu]
bench-asr-long:  ## Run long-form ASR benchmark with chunking/stride
	$(RUNNER) asr \
		--model $(MODEL) \
		--audio $(if $(AUDIO),$(AUDIO),$(LONG_AUDIO)) \
		--device $(DEVICE) \
		$(if $(CHUNK),--chunk-len $(CHUNK),) \
		$(if $(STRIDE),--stride-len $(STRIDE),) \
		$(if $(BATCH),--batch-size $(BATCH),) \
		--output

# Usage: make bench-asr-all AUDIO=data/audio/clean_speech_10s.wav [REF="text"] [DEVICE=cpu]
bench-asr-all:
ifndef AUDIO
	$(error AUDIO is required. Usage: make bench-asr-all AUDIO=audio.wav)
endif
	$(PYTHON) -c "\
from bench.runner import run_batch_asr_sweep, format_result_table; \
results = run_batch_asr_sweep('$(AUDIO)', reference='$(REF)' if '$(REF)' else None, device='$(if $(DEVICE),$(DEVICE),cpu)'); \
print(format_result_table(results))"

# Usage: make bench-report-asr
bench-report-asr:
	$(PYTHON) -c "\
from bench.runner import generate_bench_report; \
print(generate_bench_report(surface='asr'))"

# Usage: make bench-report-asr-stream
bench-report-asr-stream:
	$(PYTHON) -c "\
from bench.runner import generate_bench_report; \
print(generate_bench_report(surface='asr_stream'))"

# Usage: make bench-enhance MODEL=rnnoise NOISY=data/audio/noisy_speech_10s_20pct.wav [CLEAN=data/audio/clean_speech_10s.wav]
bench-enhance:
ifndef MODEL
	$(error MODEL is required. Usage: make bench-enhance MODEL=rnnoise NOISY=noisy.wav)
endif
ifndef NOISY
	$(error NOISY is required. Usage: make bench-enhance MODEL=rnnoise NOISY=noisy.wav)
endif
	$(PYTHON) -c "\
import json; \
from bench.runner import run_enhance_bench, save_result; \
result = run_enhance_bench('$(MODEL)', '$(NOISY)', clean_path='$(CLEAN)' if '$(CLEAN)' else None, device='$(if $(DEVICE),$(DEVICE),cpu)'); \
save_result(result); \
print(json.dumps(result, indent=2))"

# Usage: make bench-report-enhance
bench-report-enhance:
	$(PYTHON) -c "\
from bench.runner import generate_bench_report; \
print(generate_bench_report(surface='enhance'))"

# Usage: make bench-separate MODEL=demucs AUDIO=data/audio/mix.wav [DEVICE=cpu]
bench-separate:
ifndef MODEL
	$(error MODEL is required. Usage: make bench-separate MODEL=demucs AUDIO=mix.wav)
endif
ifndef AUDIO
	$(error AUDIO is required. Usage: make bench-separate MODEL=demucs AUDIO=mix.wav)
endif
	$(RUNNER) separate --model $(MODEL) --audio $(AUDIO) --device $(DEVICE) --output

bench-report-separate:  ## Generate separation benchmark report
	$(PYTHON) -c "\
from bench.runner import generate_bench_report; \
print(generate_bench_report(surface='separate'))"

# Usage: make bench-classify MODEL=clap AUDIO=data/audio/dog.wav [DEVICE=cpu]
bench-classify:
ifndef MODEL
	$(error MODEL is required. Usage: make bench-classify MODEL=clap AUDIO=dog.wav)
endif
ifndef AUDIO
	$(error AUDIO is required. Usage: make bench-classify MODEL=clap AUDIO=dog.wav)
endif
	$(RUNNER) classify --model $(MODEL) --audio $(AUDIO) --device $(DEVICE) --output

bench-report-classify:
	$(PYTHON) -c "\
from bench.runner import generate_bench_report; \
print(generate_bench_report(surface='classify'))"

# Usage: make bench-embed MODEL=clap AUDIO=data/audio/dog.wav [DEVICE=cpu]
bench-embed:
ifndef MODEL
	$(error MODEL is required. Usage: make bench-embed MODEL=clap AUDIO=dog.wav)
endif
ifndef AUDIO
	$(error AUDIO is required. Usage: make bench-embed MODEL=clap AUDIO=dog.wav)
endif
	$(RUNNER) embed --model $(MODEL) --audio $(AUDIO) --device $(DEVICE) --output

bench-report-embed:
	$(PYTHON) -c "\
from bench.runner import generate_bench_report; \
print(generate_bench_report(surface='embed'))"

# Usage: make tts MODEL=lfm2_5_audio TEXT="Hello world"
# Usage: make tts MODEL=lfm2_5_audio TEXT="Hello world" [DEVICE=cpu]
tts: bench-tts

bench-tts:
ifndef MODEL
	$(error MODEL is required. Usage: make tts MODEL=lfm2_5_audio TEXT="Hello")
endif
ifndef TEXT
	$(error TEXT is required. Usage: make tts MODEL=lfm2_5_audio TEXT="Hello")
endif
	$(RUNNER) tts --model $(MODEL) --text "$(TEXT)" --device $(DEVICE) --output

# Usage: make test-tts MODEL=lfm2_5_audio [DATASET=tts_smoke_v1]
test-tts:
	$(PYTHON) scripts/run_tts.py --model $(MODEL) --dataset $(if $(DATASET),$(DATASET),tts_smoke_v1)

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
	@echo "  HF Pro Sprint:"
	@echo "    make hf-preflight  - Check deps/tools/datasets/env for the sprint"
	@echo "    make hf-plan       - Generate sprint plan + queues"
	@echo "    make hf-run-all    - Run all queues in parallel + report"
	@echo "    make hf-report     - Aggregate ledgers into reports"
	@echo "    make hf-prefetch   - Prefetch HF Hub assets (small strategy)"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean         - Remove cache files"

# =============================================================================
# HF Pro Sprint (2026 Q1)
# =============================================================================

hf-preflight:
	$(PYTHON) scripts/hf_sprint_preflight.py --config config/hf_sprint_2026q1.yaml

hf-plan:
	$(PYTHON) scripts/hf_sprint_plan.py --config config/hf_sprint_2026q1.yaml --output-dir runs/hf_sprint_2026q1

hf-run-all:
	$(PYTHON) scripts/hf_sprint_run_all.py --preflight

hf-report:
	$(PYTHON) scripts/hf_sprint_report.py --execution-root runs/hf_sprint_2026q1/execution --out-dir runs/hf_sprint_2026q1/reports --sprint-id hf_pro_2026q1

hf-prefetch:
	$(PYTHON) scripts/hf_prefetch.py --strategy small

# =============================================================================
# Isolation (LCS-I)
# =============================================================================

# Usage: make model-venv MODEL=clap [PYTHON_VERSION=3.12]
model-venv:
ifndef MODEL
	$(error MODEL is required)
endif
	@echo "Creating venv for $(MODEL)..."
	uv venv models/$(MODEL)/venv --python $(if $(PYTHON_VERSION),$(PYTHON_VERSION),3.12)
	@echo "✓ Created models/$(MODEL)/venv"

# Usage: make model-install MODEL=clap
model-install:
ifndef MODEL
	$(error MODEL is required)
endif
	@echo "Installing requirements for $(MODEL)..."
	uv pip install -p models/$(MODEL)/venv -r models/$(MODEL)/requirements.txt
	@echo "✓ Installed requirements for $(MODEL)"

# Usage: make model-run MODEL=clap CMD="python scripts/run_bench.py ..."
model-run:
ifndef MODEL
	$(error MODEL is required)
endif
ifndef CMD
	$(error CMD is required. Example: make model-run MODEL=clap CMD="python scripts/run_bench.py ...")
endif
	. models/$(MODEL)/venv/bin/activate && $(CMD)

# Usage: make model-shell MODEL=clap
model-shell:
ifndef MODEL
	$(error MODEL is required)
endif
	. models/$(MODEL)/venv/bin/activate && $${SHELL:-/bin/bash}

# Usage: make model-diagnose MODEL=clap
model-diagnose:
ifndef MODEL
	$(error MODEL is required)
endif
	@echo "Diagnosing $(MODEL) environment..."
	@. models/$(MODEL)/venv/bin/activate && python scripts/diagnose_env.py
