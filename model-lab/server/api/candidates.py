"""
Candidate Library - Real choices for experiments.

Provides:
- UseCases: problem domains with supported steps
- Candidates: specific configs (preset + params + expected artifacts)

All candidates are executable with today's runner surfaces.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from server.api.workbench import PRESETS


router = APIRouter(prefix="/api", tags=["candidates"])


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class UseCase:
    """A problem domain with supported steps presets."""
    use_case_id: str
    title: str
    description: str
    supported_steps_presets: List[str]


@dataclass
class Candidate:
    """A specific configuration for a use case."""
    candidate_id: str
    label: str
    use_case_id: str
    steps_preset: str
    params: Dict[str, Any]
    expected_artifacts: List[str]
    description: Optional[str] = None


# ============================================================================
# REGISTRY (source of truth)
# ============================================================================

USE_CASES: Dict[str, UseCase] = {
    "meeting_smoke": UseCase(
        use_case_id="meeting_smoke",
        title="Meeting Smoke Test",
        description="Quick validation of meeting audio processing pipeline",
        supported_steps_presets=["ingest", "full"],
    ),
    "asr_smoke": UseCase(
        use_case_id="asr_smoke",
        title="ASR Smoke Test",
        description="Speech-to-text accuracy evaluation",
        supported_steps_presets=["ingest", "full"],
    ),
    "diarization_smoke": UseCase(
        use_case_id="diarization_smoke",
        title="Diarization Smoke Test",
        description="Speaker identification accuracy evaluation",
        supported_steps_presets=["full"],
    ),
}

CANDIDATES: Dict[str, Candidate] = {
    # Meeting smoke candidates
    "meeting_ingest_fast": Candidate(
        candidate_id="meeting_ingest_fast",
        label="Fast Ingest",
        use_case_id="meeting_smoke",
        steps_preset="ingest",
        params={},
        expected_artifacts=["bundle/manifest.json", "bundle/audio_normalized.wav"],
        description="Quick preprocessing only - no transcription",
    ),
    "meeting_full_default": Candidate(
        candidate_id="meeting_full_default",
        label="Full Pipeline",
        use_case_id="meeting_smoke",
        steps_preset="full",
        params={},
        expected_artifacts=[
            "bundle/manifest.json",
            "bundle/transcript.json",
            "bundle/summary.md",
            "bundle/action_items.csv",
        ],
        description="Complete pipeline with transcription, diarization, summary",
    ),
    # ASR smoke candidates
    "asr_ingest_only": Candidate(
        candidate_id="asr_ingest_only",
        label="Ingest Only",
        use_case_id="asr_smoke",
        steps_preset="ingest",
        params={},
        expected_artifacts=["bundle/manifest.json"],
        description="Audio normalization without transcription",
    ),
    "asr_full_default": Candidate(
        candidate_id="asr_full_default",
        label="Full ASR",
        use_case_id="asr_smoke",
        steps_preset="full",
        params={},
        expected_artifacts=["bundle/transcript.json"],
        description="Complete ASR with default model",
    ),
    # Diarization candidates
    "diar_full_default": Candidate(
        candidate_id="diar_full_default",
        label="Full Diarization",
        use_case_id="diarization_smoke",
        steps_preset="full",
        params={},
        expected_artifacts=["bundle/diarization.json"],
        description="Speaker diarization with default model",
    ),
}


def get_candidates_for_use_case(use_case_id: str) -> List[Candidate]:
    """Get all candidates for a use case."""
    return [c for c in CANDIDATES.values() if c.use_case_id == use_case_id]


def get_candidate(candidate_id: str) -> Optional[Candidate]:
    """Get a candidate by ID."""
    return CANDIDATES.get(candidate_id)


def get_candidate_snapshot(candidate_id: str) -> Optional[Dict[str, Any]]:
    """Get a snapshot of candidate config for reproducibility."""
    candidate = get_candidate(candidate_id)
    if not candidate:
        return None
    return {
        "candidate_id": candidate.candidate_id,
        "label": candidate.label,
        "steps_preset": candidate.steps_preset,
        "params": candidate.params.copy(),
    }


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.get("/use-cases")
def list_use_cases() -> JSONResponse:
    """List all available use cases."""
    return JSONResponse(content=[asdict(uc) for uc in USE_CASES.values()])


@router.get("/use-cases/{use_case_id}")
def get_use_case(use_case_id: str) -> JSONResponse:
    """Get a specific use case."""
    uc = USE_CASES.get(use_case_id)
    if not uc:
        raise HTTPException(status_code=404, detail="Use case not found")
    return JSONResponse(content=asdict(uc))


@router.get("/use-cases/{use_case_id}/candidates")
def list_candidates_for_use_case(use_case_id: str) -> JSONResponse:
    """List all candidates for a use case."""
    if use_case_id not in USE_CASES:
        raise HTTPException(status_code=404, detail="Use case not found")
    
    candidates = get_candidates_for_use_case(use_case_id)
    return JSONResponse(content=[asdict(c) for c in candidates])


@router.get("/candidates/{candidate_id}")
def get_candidate_endpoint(candidate_id: str) -> JSONResponse:
    """Get a specific candidate."""
    candidate = get_candidate(candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return JSONResponse(content=asdict(candidate))
