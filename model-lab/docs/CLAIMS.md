# Claims Registry (Append-Only)

Purpose: Prevent cross-agent contradictions by requiring every non-trivial claim to be recorded with evidence. Essential for model benchmarks, performance claims, and accuracy metrics.

**Supported Domains**: This is an open exploration lab. Any AI model capability is in scope:

- **Audio**: Speech recognition, synthesis, music generation, separation, enhancement, emotion, diarization, cloning, sound effects, beat detection...
- **Vision**: Classification, detection, segmentation, generation, super-resolution, style transfer, OCR, faces, poses, depth, 3D reconstruction...
- **Video**: Classification, generation, tracking, action recognition, captioning, interpolation, summarization...
- **Multimodal**: Image captioning, VQA, text-to-image, document understanding, audio-visual, embodied AI...
- **Generative**: Text, code, music, 3D, molecules, motion, anything generative...
- **Analysis**: Embeddings, similarity, clustering, anomaly detection, feature extraction...
- **Scientific**: Medical imaging, satellite, molecular, protein folding, climate, physics simulations...
- **Other**: Reinforcement learning, robotics, time series, tabular, graph neural networks...

If it's a model capability, it belongs here. Explore freely.

Rules:

- Append-only (never rewrite prior entries).
- Every claim must be labeled exactly one of: `Observed`, `Inferred`, `Unknown`.
- Claims that affect audits/reviews must be referenced in the relevant `docs/audit/*.md` artifact.
- Model performance claims MUST include: model version, test dataset, hardware, and reproducible command.

## Template

```markdown
### CLM-YYYYMMDD-### :: <Short claim title>

Date: YYYY-MM-DD
Owner: <person/agent>
Scope: <model(s)/component(s)>
Modality: <audio|vision|video|multimodal|other>
Claim: <one sentence>
Evidence type: Observed|Inferred|Unknown

Evidence:

**Command**: `<command run>`  # OR: File reference(s)

**Output**:
<paste output or cite file path + snippet anchor>

**Hardware**: <device, e.g., MPS M1 Pro, CUDA RTX 4090, CPU>
**Model version**: <e.g., whisper-large-v3, clip-vit-large, stable-diffusion-xl>
**Dataset**: <e.g., LibriSpeech, ImageNet-1k, COCO, custom>
**Metrics**: <e.g., WER, accuracy, mAP, FID, BLEU>

Interpretation: <what the evidence shows, without upgrading Inferred to Observed>

Refs:
- Ticket: TCK-YYYYMMDD-###
```

## Metrics by Domain & Task

### Audio Domain

| Task | Common Metrics |
|------|----------------|
| Speech Recognition (ASR) | WER, CER, RTF, first-token latency |
| Speech Synthesis (TTS) | MOS, RTF, naturalness, similarity |
| Voice Cloning | Speaker similarity, MOS |
| Music Generation | FAD, MOS, musicality |
| Audio Separation | SDR, SIR, SAR |
| Sound Classification | Accuracy, F1, mAP |
| Speaker Diarization | DER, JER |
| Audio Enhancement | PESQ, STOI, SI-SNR |
| Emotion Recognition | Accuracy, F1, UAR |
| Beat/Tempo Detection | F1, accuracy (±2%) |

### Vision Domain

| Task | Common Metrics |
|------|----------------|
| Image Classification | Top-1, Top-5 accuracy |
| Object Detection | mAP, mAP@50, AP per class |
| Segmentation | mIoU, pixel accuracy, Dice |
| Image Generation | FID, IS, CLIP score, KID |
| Super-Resolution | PSNR, SSIM, LPIPS |
| Style Transfer | User preference, content preservation |
| OCR / Text Extraction | Character accuracy, word accuracy |
| Face Recognition | TAR@FAR, AUC |
| Pose Estimation | PCK, mAP, OKS |
| Depth Estimation | AbsRel, RMSE, δ < 1.25 |
| 3D Reconstruction | Chamfer distance, F-score |

### Video Domain

| Task | Common Metrics |
|------|----------------|
| Video Classification | Accuracy, mAP |
| Action Recognition | Top-1, Top-5, mAP |
| Object Tracking | MOTA, MOTP, IDF1 |
| Video Generation | FVD, temporal consistency |
| Video Captioning | BLEU, METEOR, CIDEr |
| Frame Interpolation | PSNR, SSIM, LPIPS |

### Multimodal Domain

| Task | Common Metrics |
|------|----------------|
| Image Captioning | BLEU, CIDEr, METEOR, SPICE |
| Visual QA | Accuracy, consistency |
| Text-to-Image | FID, CLIP score, user preference |
| Image-to-Text | BLEU, accuracy |
| Document Understanding | Accuracy, F1 |
| Audio-Visual | Task-specific |

### Generative Domain

| Task | Common Metrics |
|------|----------------|
| Text Generation | Perplexity, BLEU, human eval |
| Code Generation | Pass@k, functional correctness |
| 3D Generation | FID-3D, user study |
| Molecule Generation | Validity, novelty, diversity |

### Common Metrics (All Models)

| Metric | Description |
|--------|-------------|
| Inference latency | Time per sample (ms) |
| Throughput | Samples per second |
| Memory usage | Peak VRAM/RAM (GB) |
| RTF | Real-time factor (for streaming) |
| Model size | Parameters (M/B) |
| FLOPS | Computational cost |

---

### CLM-20260205-001 :: Whisper Large V3 WER on LibriSpeech

Date: 2026-02-05
Owner: Model Lab
Scope: models/whisper
Modality: audio
Claim: Whisper large-v3 achieves ~2.5% WER on LibriSpeech test-clean
Evidence type: Observed

Evidence:

**Command**: `PYTHONPATH=. python -m harness.run --model whisper-large-v3 --dataset librispeech-test-clean`

**Output**:
```
WER: 2.47%
CER: 0.89%
RTF: 0.15x (MPS)
```

**Hardware**: Apple M1 Pro, 16GB, MPS backend
**Model version**: openai/whisper-large-v3
**Dataset**: LibriSpeech test-clean (2620 samples)
**Metrics**: WER, CER, RTF

Interpretation: Observed — Results from reproducible benchmark run. Consistent with OpenAI published metrics.

Refs:
- Ticket: N/A (baseline benchmark)

---
