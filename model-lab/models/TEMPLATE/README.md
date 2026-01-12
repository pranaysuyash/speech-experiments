# Model Template

Use this template to onboard new models into the arsenal.

## Files to Create

1. `config.yaml` - Model configuration
2. `README.md` - Documentation (copy this template)
3. Complete `onboarding_checklist.md`

## Config Structure

```yaml
model_type: your_model_id
model_name: "org/model-name"
version: "1.0.0"

inference:
  compute_type: float16
  dtype: float16
  max_tokens: 256

# For CLI adapters (whisper.cpp, etc)
whisper_cpp:
  bin_path: /path/to/binary
  model_path: /path/to/model.gguf
```

## Loader Registration

Add to `harness/registry.py`:

```python
def load_your_model(config: Dict[str, Any], device: str) -> Bundle:
    # Implementation here
    return {
        "model_type": "your_model_id",
        "device": device,
        "capabilities": ["asr"],  # list what it can do
        "asr": {"transcribe": transcribe_fn},
        "raw": {"model": model}
    }

ModelRegistry.register_loader(
    "your_model_id",
    load_your_model,
    "Description here",
    status=ModelStatus.EXPERIMENTAL,
    version="1.0.0",
    capabilities=["asr"],
    hardware=["cpu", "mps", "cuda"],
)
```

## Quick Test

```bash
make model-info MODEL=your_model_id
make asr MODEL=your_model_id DATASET=llm_primary
```
