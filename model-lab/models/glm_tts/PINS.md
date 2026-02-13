# GLM-TTS Dependency Pins

Strict canonical source: zai-org/GLM-TTS only.

## Repository Pin

| Field | Value |
|-------|-------|
| Repository | https://github.com/zai-org/GLM-TTS.git |
| Commit | `c5dc7aecc3b4032032d631b271e767893984f821` |
| Date | 2025-12-17 10:42:57 +0800 |

## Model Weights Source

| Field | Value |
|-------|-------|
| HuggingFace Repo | zai-org/GLM-TTS |
| Download Command | `huggingface-cli download zai-org/GLM-TTS --local-dir models/glm_tts/ckpt` |

## Patches

| Patch | File | Description |
|-------|------|-------------|
| 0001 | `cosyvoice/cli/frontend.py` | Replace `.cuda()` with dynamic device selection |
| 0002 | `cosyvoice/utils/file_utils.py` | Replace `.cuda()` with dynamic device selection |
| 0003 | `utils/file_utils.py` | Replace `.cuda()` with dynamic device selection |
| 0004 | `utils/yaml_util.py` | Replace `.cuda()` with dynamic device selection |

## Python Dependencies

| Package | Source | Version |
|---------|--------|---------|
| torch | PyPI | 2.3.1 |
| transformers | PyPI | 4.57.3 |
| pynini | PyPI | 2.1.7 |

## System Dependencies

| Dependency | Install Command | Verification |
|------------|-----------------|--------------|
| OpenFST (macOS) | `brew install openfst` | `pkg-config --exists openfst` or `/opt/homebrew/lib/libfst.dylib` |

## Verification

```bash
cd models/glm_tts/repo
git log -1 --format="%H"
# Expected: c5dc7aecc3b4032032d631b271e767893984f821
```
