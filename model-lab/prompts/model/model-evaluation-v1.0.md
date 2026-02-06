# Model Evaluation Prompt

**Version**: 1.0  
**Purpose**: Systematically evaluate AI models with reproducible benchmarks.

**This is an open exploration lab.** Any AI model capability is in scope. Don't limit yourself to common tasks - explore music generation, audio separation, pose estimation, molecule generation, or anything a model can do.

## Use When

- Adding a new model to the registry
- Comparing model performance
- Exploring a new model capability
- Verifying model claims
- Creating benchmark documentation

## Non-Negotiable Rules

1. **Reproducible**: Every result must include command to reproduce
2. **Hardware documented**: Always specify MPS/CUDA/CPU
3. **Dataset/input specified**: Always specify test data
4. **Claims registered**: Results go in docs/CLAIMS.md
5. **Multiple runs**: Report average of 3+ runs for metrics
6. **Explore freely**: Don't limit to predefined categories

## Inputs

- Model name and version
- Domain (audio/vision/video/multimodal/generative/scientific/other)
- Task type (be specific: "music generation", "protein folding", "audio separation", etc.)
- Target hardware (MPS/CUDA/CPU)
- Test dataset or input samples
- Metrics to measure (see docs/CLAIMS.md for extensive metrics reference)

## Steps

### 1. Environment Verification

```bash
# Check Python version
python --version

# Check PyTorch and device
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS: {torch.backends.mps.is_available()}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check available memory
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().total / 1e9:.1f} GB')"
```

### 2. Model Loading Test

```bash
# Verify model loads
PYTHONPATH=. python -c "from harness.models import load_model; m = load_model('<model-name>'); print('Model loaded successfully')"

# Check model size
PYTHONPATH=. python -c "from harness.models import load_model; m = load_model('<model-name>'); print(f'Parameters: {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M')"
```

### 3. Inference Test

```bash
# Single sample test
PYTHONPATH=. python -m harness.run --model <model-name> --input inputs/test.wav --quick

# Verify output format
# Check for errors, warnings
```

### 4. Benchmark Run

```bash
# Full benchmark (3 runs)
for i in 1 2 3; do
    PYTHONPATH=. python -m harness.run \
        --model <model-name> \
        --dataset <dataset> \
        --output runs/benchmark_run_$i.json
done
```

### 5. Metrics Collection

**See `docs/CLAIMS.md` for comprehensive metrics by domain.**

Choose metrics appropriate for your task. Examples:

**Audio Tasks**
- Speech: WER, CER, RTF, MOS
- Music: FAD, musicality, MOS
- Separation: SDR, SIR, SAR
- Enhancement: PESQ, STOI

**Vision Tasks**
- Classification: Top-1, Top-5
- Detection: mAP, AP@50
- Generation: FID, IS, CLIP
- 3D: Chamfer, F-score

**Video Tasks**
- Classification: accuracy, mAP
- Tracking: MOTA, IDF1
- Generation: FVD

**Generative Tasks**
- Text: perplexity, BLEU
- Code: Pass@k
- 3D/Molecules: domain-specific

**Universal Metrics**
- Inference latency (ms)
- Memory usage (peak VRAM/RAM)
- Throughput (samples/sec)
- Model size (parameters)

**Custom Metrics**: Define your own if the task is novel. Document how it's computed.

### 6. Register Claims

Add to `docs/CLAIMS.md`:
```markdown
### CLM-YYYYMMDD-### :: [Model] [Metric] on [Dataset]

Date: YYYY-MM-DD
Owner: [agent/person]
Scope: models/[model-name]
Claim: [Model] achieves [metric value] on [dataset]
Evidence type: Observed

Evidence:

**Command**: `PYTHONPATH=. python -m harness.run --model <model-name> --dataset <dataset>`

**Output**:
```
[paste output]
```

**Hardware**: [device]
**Model version**: [exact version]
**Dataset**: [dataset name and size]

Interpretation: [analysis]

Refs:
- Ticket: TCK-YYYYMMDD-###
```

## Output Format

```markdown
# Model Evaluation Report: [Model Name]

**Date**: YYYY-MM-DD
**Evaluator**: [name/agent]
**Hardware**: [device details]

## Model Information

| Property | Value |
|----------|-------|
| Name | [name] |
| Version | [version] |
| Modality | [audio/vision/video/multimodal] |
| Task | [ASR/TTS/classification/detection/generation/etc] |
| Parameters | [size] |
| Source | [HuggingFace/etc] |

## Environment

| Component | Version |
|-----------|---------|
| Python | [version] |
| PyTorch | [version] |
| Device | [MPS/CUDA/CPU] |
| RAM | [size] |

## Benchmark Results

### Dataset: [name]

| Metric | Run 1 | Run 2 | Run 3 | Average |
|--------|-------|-------|-------|---------|
| WER | | | | |
| RTF | | | | |
| Memory | | | | |

## Reproducibility

```bash
# Command to reproduce
PYTHONPATH=. python -m harness.run --model [model] --dataset [dataset]
```

## Observations

[qualitative notes, edge cases, issues]

## Recommendation

- [ ] Add to registry
- [ ] Needs more testing
- [ ] Not recommended (reason)

## Claims Registered

- CLM-YYYYMMDD-### - [claim summary]
```

## Stop Condition

Stop when:
- All metrics collected with 3+ runs
- Results registered in CLAIMS.md
- Report documented
- Reproducibility command verified
