# Audit: ACE-Step 1.5 + KugelAudio-0 (Reported) — 2026-02-05

Status: **Reported** (not reproduced in Model‑Lab runs)  
Scope: Quick research capture + catalog entries (no benchmarking).  

## Why this exists

User request (2026-02-05): “check these out as well: Ace Step 1.5 & Kugel Audio 0”.

This audit captures **what these models claim to be**, **where they live**, and **what’s required to run them**, without promoting any claim to **Observed**.

## Model: ACE-Step 1.5 (music generation)

### What it is (Reported)

- A text-to-music (“Text2Music”) foundation model entry published as `ACE-Step/Ace-Step1.5` on Hugging Face. (Reported)
- Model card claims:
  - Consumer-hardware friendly (“<4GB VRAM”). (Reported)
  - Fast generation claims (A100 / RTX 3090). (Reported)
  - Prompt adherence across “50+ languages”. (Reported)

### Primary sources (Reported)

- Hugging Face model card: `ACE-Step/Ace-Step1.5` — https://huggingface.co/ACE-Step/Ace-Step1.5
- GitHub repository: `ace-step/ACE-Step` — https://github.com/ace-step/ACE-Step
- Third-party vendor blog reference (AMD) summarizing claims — https://rocm.blogs.amd.com/artificial-intelligence/ACE-Step/README.html

### Licensing note (needs explicit verification)

- HF model card lists **MIT** license. (Reported)
- GitHub repo displays **Apache-2.0** license for the repository. (Reported)

Action: treat license as **Unknown for “weights vs code vs tooling”** until a legal review confirms scope.

### Practical lab implications (Inferred)

- This is a **music generation** model family, not directly comparable to ASR/TTS metrics in `PERFORMANCE_RESULTS.md`.
- If we want to evaluate it in Model‑Lab, we likely need:
  - A generation harness (duration, prompt set, seed handling).
  - Subjective evaluation or FAD-like proxies (if available), plus latency/throughput.

## Model: KugelAudio-0-Open (“Kugel Audio 0”) (TTS + voice cloning)

### What it is (Reported)

- A 7B-parameter open-weight TTS model card on Hugging Face: `kugelaudio/kugelaudio-0-open`. (Reported)
- Model card claims:
  - European-language focus + voice cloning. (Reported)
  - ~19GB VRAM requirement for inference. (Reported)
  - “AR + Diffusion” architecture with a Qwen2.5-7B backbone. (Reported)
  - A hosted API (kugelaudio.com) and an example Python client snippet. (Reported)

### Primary source (Reported)

- Hugging Face model card: `kugelaudio/kugelaudio-0-open` (includes license + hardware + language notes) — https://huggingface.co/kugelaudio/kugelaudio-0-open

### Licensing (Reported)

- HF model card indicates **MIT** license. (Reported)

### Safety / consent (Inferred)

- Any “voice cloning” capability should be treated as **high consent risk** until we formalize:
  - allowed use-cases,
  - provenance/consent checks for reference audio,
  - explicit logging and watermarking strategy (if available).

## Follow-ups (optional; out of scope for this ticket)

- Add these models to a generation/TTS benchmark harness and create **Observed** runs under `model-lab/runs/**`.
- Expand the catalog schema for generation models (duration/steps/sample-rate/conditioning controls).
