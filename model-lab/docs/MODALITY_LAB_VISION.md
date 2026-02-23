# Modality Lab Vision

Last updated: 2026-02-22
Owner: model-lab

## North Star
Evolve `model-lab` from an audio-focused benchmark harness into a full modality testing and model-finder platform that can rank, recommend, and gate models across:
- audio
- vision
- video
- text
- multimodal/realtime
- agentic/tool-using workflows

## Product Direction
1. Unified Model Finder
- Query by task, latency, cost, quality, hardware, and deployment constraints.
- Return ranked model+pipeline recommendations with reproducible evidence.

2. Evaluation OS
- Standardized run contracts across modalities.
- Common artifact schema (input fingerprint, pipeline config, metrics, provenance, gates).
- Deterministic preprocessing/postprocessing and regression detection.

3. Production Promotion System
- Lifecycle states: `experimental -> candidate -> production`.
- Non-regression and quality gates before promotion.
- Traceable run lineage and audit-ready reports.

## Core Building Blocks
1. Modality Registry
- Capability taxonomy (example: `asr`, `diarization`, `ocr`, `vqa`, `captioning`, `tool_use`).
- Runtime compatibility matrix (cpu/mps/cuda/onnx/mlx/gguf/api).

2. Scenario Library
- Task packs with inputs, references, acceptance criteria, and failure tags.
- Versioned datasets and scenario definitions.

3. Metrics/Gates Layer
- Per-modality objective metrics.
- Cross-modality operational metrics (latency, cost, robustness, stability).
- Business gates mapped to use-case requirements.

4. Recommendation Engine
- Multi-objective ranking (`fast`, `balanced`, `high_quality`, `realtime`).
- Explainable selection (`why this model/workflow`).

5. UX Layer
- Workbench for ad-hoc and compare flows.
- Finder page for constraints-based model selection.
- Evidence browser for run-to-run analysis.

## Immediate Architecture Path
1. Stabilize audio as reference implementation.
2. Introduce generalized `contracts_v2` for modality-agnostic run artifacts.
3. Add first non-audio vertical (recommended: document/vision tasks).
4. Add recommendations API + UI backed by measured runs.
5. Add automated model-ingestion and periodic benchmark runs.

## Risks and Controls
1. Contract drift across runners
- Control: contract parity tests and schema validation at write time.

2. Metric inconsistency across modalities
- Control: centralized metric interfaces and versioned metric policies.

3. Unbounded growth of models/workflows
- Control: promotion gates, scenario coverage requirements, and archive policy.

## Success Criteria
1. A single run contract works across at least 3 modalities.
2. Finder recommendations are evidence-backed and reproducible.
3. New model onboarding is mostly automated with regression checks.
4. Teams can choose a model/workflow from constraints in minutes, not days.
