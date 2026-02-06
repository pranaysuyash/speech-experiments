---
name: Model Request
about: Request addition of a new AI model (audio, vision, video, multimodal)
title: '[MODEL] '
labels: model
assignees: ''
---

## Model Information
- **Model name**: 
- **Modality**: (Audio/Vision/Video/Multimodal)
- **Task type**: (ASR/TTS/Classification/Detection/Generation/Other)
- **Source**: (HuggingFace/GitHub/Other)
- **Model URL**: 

## Why Add This Model?
<!-- What value does this model add to the comparison lab? -->

## Technical Requirements
- **Framework**: (transformers/torch/diffusers/other)
- **Hardware**: (MPS compatible? CUDA required? CPU only?)
- **Memory**: (estimated VRAM/RAM)
- **Dependencies**: 

## Benchmark Expectations
<!-- What benchmarks should we run? Check applicable or add custom -->

**Audio**
- [ ] WER/CER (speech)
- [ ] MOS (synthesis quality)
- [ ] SDR/SIR (separation)
- [ ] FAD (music)

**Vision**
- [ ] Accuracy (classification)
- [ ] mAP (detection)
- [ ] mIoU (segmentation)
- [ ] FID/CLIP (generation)
- [ ] PSNR/SSIM (restoration)

**Video**
- [ ] Accuracy/mAP
- [ ] MOTA (tracking)
- [ ] FVD (generation)

**Other Tasks**
- [ ] Pass@k (code)
- [ ] BLEU/CIDEr (captioning)
- [ ] Custom: ___________

**Universal**
- [ ] Inference latency
- [ ] Memory usage
- [ ] Throughput
- [ ] Model size 

## Integration Notes
<!-- Any special considerations for integrating this model -->

## Priority
- [ ] High (fills critical gap)
- [ ] Medium (nice to have)
- [ ] Low (exploratory)
