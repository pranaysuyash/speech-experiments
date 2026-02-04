# Comprehensive Product Audit: Model Lab

**Created:** 2026-02-04
**Auditor:** GitHub Copilot (senior staff engineer + product architect + technical program manager)
**Coverage:** Deep-read: README.md, QUICKSTART.md, server/main.py, client/package.json, pyproject.toml, key docs (20%). Skimmed: API files, harness modules, test results (50%). Skipped: node_modules, __pycache__, large data files (30%).

## Executive Summary

**What it is now:** Model Lab is a local testing framework for comparing AI speech models (ASR, TTS, Chat) using shared harnesses, Jupyter notebooks, and a React/FastAPI web UI. It supports models like LFM2.5-Audio, Whisper, and Faster-Whisper, producing automated production readiness scorecards. Current production-ready models: Faster-Whisper (A+), Whisper (A); LFM2.5 (C, hallucinations).

**What it aims to become:** Scalable model evaluation platform with more models, better UI, and production deployment support (from docs/ROADMAP.md and testing expansions).

**Top 5 Risks:**
1. **AI-001:** Model hallucinations disqualify LFM2.5 for production (137.8% WER, severe hallucinations).
2. **SEC-001:** No authentication on API endpoints (public access to model runs/results).
3. **ARCH-001:** Monolithic backend with mixed concerns (API, services, logic in server/).
4. **PERF-001:** No model caching; slow startups (32s for LFM2.5).
5. **DATA-001:** No database; results as JSON files (scalability limits).

**Top 5 Leverage Opportunities:**
1. Expand to more models (TTS/Chat fully, new architectures).
2. Add model versioning and A/B testing UI.
3. Integrate with cloud deployment (Colab, AWS).
4. Implement hallucination detection/mitigation.
5. Add real-time metrics dashboard.

## Product & Architecture Overview

**Components:**
- **Frontend:** React/TypeScript/Vite app (client/) with routes for runs, workbench, experiments, candidates, results, findings.
- **Backend:** FastAPI server (server/) with routers for runs, results, workbench, experiments, candidates, lifecycle, pipelines, WS runs.
- **Harness:** Python modules (harness/) for shared testing logic (audio_io, metrics_asr, timers, registry).
- **Models:** Isolated folders (models/) with configs and notebooks (e.g., lfm2_5_audio/, whisper/).
- **Data:** Test datasets (data/), results (runs/), comparisons (compare/).
- **Testing:** Jupyter notebooks for experiments, pytest for unit tests.

**Boundaries:**
- User ↔ Frontend (localhost:5173) ↔ Backend (localhost:8000) ↔ Harness/Model execution ↔ Results storage.
- No external APIs; local model inference.

**Key Modules & Responsibilities:**
- server/api/: REST endpoints for lab operations.
- server/services/: Business logic (runs_index, safe_files, results_v1).
- harness/: Reusable testing utilities.
- models/*/notebooks/: Experiment scripts.

**Deployment Model:** Local development with uvicorn/FastAPI reload. No production deployment config visible (no Docker, Helm).

## Audit Findings

### ARCH-001: Monolithic Backend Architecture
- **Severity:** High
- **Confidence:** High
- **Evidence:** server/ contains mixed API, services, and logic (e.g., api/runs.py has 877 lines with business logic). No clear separation (controllers/services/repos).
- **Impact:** Maintainability low; hard to scale/modify.
- **Fix:** Refactor to layered architecture (API → Services → Repos). Effort: M (2-4 weeks). Risk: Low.
- **Effort:** M

### SEC-001: No Authentication on API Endpoints
- **Severity:** Blocker
- **Confidence:** High
- **Evidence:** server/main.py has no auth middleware; endpoints like /api/runs are public (grep shows no auth/Login in server/).
- **Impact:** Unauthorized access to model runs/results; data exposure.
- **Fix:** Add JWT/OAuth middleware. Effort: M (1-2 weeks). Risk: Med (integration testing).
- **Effort:** M

### PERF-001: No Model Caching; Slow Startups
- **Severity:** Med
- **Confidence:** High
- **Evidence:** Test results show 32s for LFM2.5 (0.196x RTF); no caching in harness/registry.py (models loaded fresh per run).
- **Impact:** Poor UX for repeated tests; resource waste.
- **Fix:** Implement model LRU cache in registry.py. Effort: S (1 week). Risk: Low.
- **Effort:** S

### DEVEX-001: Incomplete CI/CD Pipeline
- **Severity:** Med
- **Confidence:** Med
- **Evidence:** .github/workflows/ exists (ci.yml, etc.), but no full pipeline (e.g., no deployment, only tests). Makefile has build but no deploy.
- **Impact:** Manual deployments; no automated releases.
- **Fix:** Add deployment workflows (Docker, staging). Effort: M (2 weeks). Risk: Med.
- **Effort:** M

### UX-001: Frontend Lacks Error Handling and Accessibility
- **Severity:** Low
- **Confidence:** Med
- **Evidence:** App.tsx has basic routing; no error boundaries beyond RunDetailErrorBoundary. No a11y checks (no aria labels visible).
- **Impact:** Poor error UX; accessibility issues.
- **Fix:** Add global error boundaries, loading states, WCAG compliance. Effort: S (1 week). Risk: Low.
- **Effort:** S

### DATA-001: No Database; JSON File Storage
- **Severity:** High
- **Confidence:** High
- **Evidence:** Results stored as JSON in runs/ (e.g., runs/lfm2_5_audio/asr/2024-01-08_12-34-56.json). No DB in pyproject.toml.
- **Impact:** Scalability limits; no queries/concurrency.
- **Fix:** Add SQLite/PostgreSQL for runs/results. Effort: L (4-6 weeks). Risk: High (data migration).
- **Effort:** L

### OBS-001: Limited Observability; No Metrics/Tracing
- **Severity:** Med
- **Confidence:** High
- **Evidence:** Logging in server/main.py (uvicorn access logs), but no Prometheus/metrics. No tracing (e.g., no OpenTelemetry).
- **Impact:** Hard to debug production issues.
- **Fix:** Add metrics endpoints, Jaeger tracing. Effort: M (2 weeks). Risk: Med.
- **Effort:** M

### TEST-001: Low Test Coverage and Determinism
- **Severity:** Med
- **Confidence:** Med
- **Evidence:** pytest.ini exists; markers for real_e2e/slow. But coverage unknown (no coverage reports); harness tests may not cover all.
- **Impact:** Bugs in production.
- **Fix:** Add coverage reporting, deterministic fixtures. Effort: S (1 week). Risk: Low.
- **Effort:** S

### AI-001: No Hallucination Detection/Mitigation
- **Severity:** Blocker
- **Confidence:** High
- **Evidence:** Test results show LFM2.5 hallucinations (137.8% WER, repetitive output). No grounding/evals in harness/ (docs/DECISION_SEMANTICS.md notes hallucinations disqualifying, but no code controls).
- **Impact:** Unusable outputs for affected models.
- **Fix:** Add hallucination detectors (e.g., perplexity checks, fact-checking). Effort: M (2-3 weeks). Risk: High (AI complexity).
- **Effort:** M

### DEP-001: Dependency Vulnerabilities Not Scanned
- **Severity:** Med
- **Confidence:** Med
- **Evidence:** pyproject.toml has many deps (torch, transformers); no Dependabot/Snyk visible in .github/.
- **Impact:** Security vulnerabilities.
- **Fix:** Add dependency scanning to CI. Effort: S (0.5 week). Risk: Low.
- **Effort:** S

## Prioritized Roadmap

### 0–2 Weeks: Stabilize (Fix Blockers)
- SEC-001: Add basic API auth (JWT). Owner: Backend Engineer. Acceptance: Endpoints require token. Dependencies: None. Risk: Breaks existing clients (mitigate with optional auth flag).
- AI-001: Implement basic hallucination filter (reject >50% WER). Owner: ML Engineer. Acceptance: LFM outputs filtered. Dependencies: Harness updates. Risk: False positives.
- ARCH-001: Extract services layer. Owner: Backend Engineer. Acceptance: server/services/ separated. Dependencies: None. Risk: Refactor downtime.

### 2–6 Weeks: Build Leverage (Core Improvements)
- PERF-001: Add model caching. Owner: Backend Engineer. Acceptance: Startup <10s. Dependencies: Registry refactor. Risk: Memory usage.
- DATA-001: Add SQLite for results. Owner: Backend Engineer. Acceptance: Queries work. Dependencies: Migration script. Risk: Data loss (backup first).
- OBS-001: Add basic metrics. Owner: DevOps Engineer. Acceptance: /metrics endpoint. Dependencies: Prometheus client. Risk: None.

### 6–12 Weeks: Scale and Polish (Advanced Features)
- DEVEX-001: Full CI/CD with Docker deploy. Owner: DevOps Engineer. Acceptance: Staging env. Dependencies: Cloud account. Risk: Cost.
- UX-001: Improve error handling/accessibility. Owner: Frontend Engineer. Acceptance: WCAG AA. Dependencies: None. Risk: None.
- TEST-001: 80% coverage. Owner: QA Engineer. Acceptance: Coverage report. Dependencies: None. Risk: None.
- DEP-001: Automated vuln scanning. Owner: Security Engineer. Acceptance: Weekly reports. Dependencies: CI update. Risk: None.

## Research Appendix

**Competitive/Landscape Notes:**
- Hugging Face Model Hub: Similar for model testing, but cloud-based; local labs like this are rare, often custom (e.g., OpenAI internal).
- MLflow: Experiment tracking, but not speech-specific; this lab adds domain expertise.
- Weights & Biases: Model comparison, but no local inference; this is more integrated.

**Best-Practice References:**
- OWASP AI Security: Hallucination risks (owasp.org); grounding techniques (e.g., RAG).
- ML Model Serving: TorchServe for caching (pytorch.org).
- Observability: Prometheus for metrics (prometheus.io); OpenTelemetry for tracing.

**Library/Framework Doc Citations:**
- FastAPI Security: JWT auth (fastapi.tiangolo.com/tutorial/security/).
- PyTorch Caching: Model loading optimizations (pytorch.org/docs).
- React Error Boundaries: Handling errors (react.dev/reference/react/Component#catching-rendering-errors).

**Security/Privacy References:**
- API Auth: OAuth 2.0 best practices (oauth.net).
- Data Storage: JSON vs DB tradeoffs (sqlite.org); GDPR for PII (gdpr.eu).

---

(End of audit)