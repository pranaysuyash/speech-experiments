# Model Lab - Project Rules
# These are non-negotiable policies. Enforce with tooling.

## 1. Environment Rules

### uv is mandatory
- All installs via `uv add`, `uv remove`, `uv sync`
- **No `pip install` in this repo. Ever.**
- Commit `uv.lock`

### Single Python version
- Pin in `.python-version` and `pyproject.toml` `requires-python`
- Currently: `3.12`

### No running Python outside uv
- Never `python foo.py`
- Always: `uv run python foo.py` or `make <target>`

### Jupyter kernel
- Use the uv-registered kernel
- Run `make setup-kernel` if notebooks fail

---

## 2. Repo Structure Rules

### data/ is local, golden/ is versioned
- `data/` ignored by default (big files)
- `data/golden/` tracked (small, curated, stable)

### runs/ are artifacts
- `runs/` gitignored by default
- Only commit: `summary.json`, `model_card.json`

### Notebooks are demos, scripts are truth
- No business logic in notebooks
- Official tests must have script entry point

---

## 3. File Limits

### Hard size limit: 2 MB
- Anything bigger → `data/` (ignored) or external storage

### No binaries in repo
- No model weights, no `.gguf`, no compiled binaries
- Store paths in config with download instructions

---

## 4. Code Quality

### Formatting + linting are not optional
- `ruff format` + `ruff check`
- Pre-commit enforces this

### Logging, not print
- In harness/scripts: use `logging`
- Only notebooks can print

### Config is YAML
- All model params in `models/<model>/config.yaml`
- Scripts accept `--config`

---

## 5. Execution Entry Points

Use Makefile targets:
```bash
make setup          # uv sync
make lint           # ruff format + check
make test           # pytest
make lab            # jupyter lab
make asr MODEL=whisper DATASET=llm_primary
```

---

## 6. Secrets

### Never commit secrets
- `.env` is gitignored
- Use `.env.example` as template
- All API keys from env vars only

---

## 7. Bundle Contract v1 (Model Loaders)

Every loader returns:
```python
{
    "model_type": str,
    "device": str,
    "capabilities": ["asr", ...],
    "asr": {"transcribe": callable},
}
```

Runner only calls `bundle["asr"]["transcribe"]()`. Never raw model methods.

---

## 8. Golden Set Requirements

Minimum 3-5 test cases with:
- Ground truth text
- Tags: `[technical, numbers, noisy, code_switch]`
- Expected language

---

## Enforcement

- Pre-commit: ruff, file size, no `pip install`
- CI: same checks
- Merge blocked without model card for new models
