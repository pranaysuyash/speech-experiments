# Model Addition Prompt

**Version**: 1.0  
**Purpose**: Add a new AI model to the model-lab registry.

**This is an open exploration lab.** Any AI model is welcome - mainstream or experimental. Music generators, protein folders, audio separators, depth estimators, motion capture, molecule designers, anything.

## Use When

- Adding a new model for comparison
- Exploring a new capability area
- Integrating a model from HuggingFace, GitHub, papers, or anywhere
- Creating a new model adapter

## Non-Negotiable Rules

1. **Ticket first**: Create worklog ticket before implementation
2. **Evaluation required**: Run evaluation before marking complete
3. **Registry update**: Add to model registry with metadata
4. **Documentation**: Update relevant docs with model info
5. **Claims registered**: Performance claims go in CLAIMS.md
6. **Explore freely**: No artificial limits on what models we add

## Inputs

- Model name and source URL
- Domain (audio/vision/video/multimodal/generative/scientific/robotics/other)
- Task type (be specific: "music generation", "depth estimation", "molecule generation", etc.)
- Target hardware compatibility
- Expected capabilities

## Steps

### 1. Create Ticket

```markdown
### TCK-YYYYMMDD-### :: Add [Model Name] to registry

Type: [MODEL]
Owner: [agent/person]
Created: YYYY-MM-DD
Status: **OPEN**
Modality: [audio/vision/video/multimodal]
Task: [ASR/TTS/classification/detection/generation/etc]

Scope contract:
- In-scope: Add model adapter, registry entry, basic tests
- Out-of-scope: Extensive optimization, streaming support
- Behavior change allowed: NO (additive only)

Acceptance Criteria:
- [ ] Model loads on target hardware
- [ ] Inference produces valid output
- [ ] Benchmark results documented
- [ ] Registry entry complete
- [ ] CLAIMS.md updated
```

### 2. Research Model

```bash
# Check model documentation
# Note: dependencies, hardware requirements, API

# Check existing adapters for patterns
ls harness/adapters/
cat harness/adapters/whisper_adapter.py | head -50
```

### 3. Create Adapter

Create `harness/adapters/<model_name>_adapter.py`:

```python
"""
Adapter for [Model Name]

Source: [URL]
Domain: [audio/vision/video/generative/scientific/etc]
Task: [specific task - be descriptive]
Hardware: [MPS/CUDA/CPU compatibility]
"""

from typing import Optional, Union, Any
import torch

class ModelNameAdapter:
    """Adapter for [Model Name]."""
    
    def __init__(self, device: str = "auto"):
        self.device = self._resolve_device(device)
        self.model = None
        self.processor = None
    
    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device
    
    def load(self):
        """Load model and processor."""
        # Implementation
        pass
    
    def __call__(self, input_data: Any, **kwargs) -> Any:
        """
        Primary inference method.
        
        Define input/output based on your model's task:
        - Audio ASR: audio_path -> text
        - Music gen: prompt -> audio
        - Image gen: prompt -> image
        - Separation: audio -> dict of stems
        - Detection: image -> list of boxes
        - Depth: image -> depth_map
        - Molecules: smiles -> properties
        - Pose: image -> keypoints
        - etc.
        """
        pass
    
    # Add task-specific convenience methods as needed:
    
    # def transcribe(self, audio_path: str) -> str:
    # def generate_music(self, prompt: str, duration: float) -> bytes:
    # def separate(self, audio_path: str) -> dict:
    # def estimate_depth(self, image_path: str) -> np.ndarray:
    # def detect_poses(self, image_path: str) -> list:
    # def generate_molecule(self, properties: dict) -> str:
    # def encode(self, data: Any) -> np.ndarray:  # embeddings
    # ... whatever your model does
```

### 4. Add Registry Entry

Update model registry:

```python
MODEL_REGISTRY["model-name"] = {
    "name": "Model Display Name",
    "domain": "audio",  # audio, vision, video, multimodal, generative, scientific, other
    "task": "music_generation",  # be specific: "music_generation", "depth_estimation", "audio_separation", etc.
    "adapter": "model_name_adapter",
    "source": "huggingface/model-name",
    "hardware": ["mps", "cuda", "cpu"],
    "parameters": "1.5B",
    "description": "Brief description of what this model does",
    "capabilities": ["generate", "condition"],  # what it can do
    "inputs": ["text", "melody"],  # what it accepts
    "outputs": ["audio"],  # what it produces
}
```

### 5. Add Tests

Create `tests/models/test_<model_name>.py`:

```python
import pytest
from harness.adapters.model_name_adapter import ModelNameAdapter

@pytest.mark.slow
def test_model_loads():
    adapter = ModelNameAdapter()
    adapter.load()
    assert adapter.model is not None

@pytest.mark.slow  
def test_inference():
    adapter = ModelNameAdapter()
    adapter.load()
    result = adapter.transcribe("inputs/test.wav")
    assert isinstance(result, str)
    assert len(result) > 0
```

### 6. Run Evaluation

Follow `prompts/model/model-evaluation-v1.0.md` for full evaluation.

### 7. Update Documentation

- Add to README.md model list
- Update docs/MODEL_REGISTRY_MASTER.md
- Add claims to docs/CLAIMS.md

## Output Format

### Completion Checklist

```markdown
## Model Addition Complete: [Model Name]

**Ticket**: TCK-YYYYMMDD-###
**Date**: YYYY-MM-DD

### Files Created/Modified

- [ ] `harness/adapters/<model>_adapter.py` - Adapter
- [ ] `harness/registry.py` - Registry entry
- [ ] `tests/models/test_<model>.py` - Tests
- [ ] `docs/MODEL_REGISTRY_MASTER.md` - Documentation
- [ ] `docs/CLAIMS.md` - Performance claims

### Verification

- [ ] Model loads: `python -c "from harness.adapters... "`
- [ ] Inference works: `python -m harness.run --model <model> --quick`
- [ ] Tests pass: `pytest tests/models/test_<model>.py`
- [ ] Benchmark complete: [link to results]

### Performance Summary

| Metric | Value |
|--------|-------|
| WER | [value] |
| RTF | [value] |
| Memory | [value] |
| Hardware | [device] |
```

## Stop Condition

Stop when:
- Adapter implemented and tested
- Registry entry complete
- Evaluation documented
- Claims registered
- Ticket marked COMPLETED
