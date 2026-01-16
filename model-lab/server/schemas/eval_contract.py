"""
Evaluation contract types and schema for model-lab.

This module defines the eval.json schema that is written per-run
to provide a stable contract for metrics, checks, and findings.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class Check:
    """A single check with pass/fail status and severity."""
    name: str
    passed: bool
    severity: str  # "info" | "warn" | "fail"
    message: str
    evidence_paths: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Finding:
    """A finding represents an issue or observation about the run."""
    finding_id: str  # Stable ID for aggregation
    severity: str  # "low" | "medium" | "high"
    category: str  # "asr" | "diarization" | "alignment" | "system"
    title: str
    details: str
    evidence_paths: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreCard:
    """A scored metric with provenance."""
    name: str  # e.g., "artifact_completeness"
    label: str  # Human readable label
    type: str  # "proxy" or "llm_judge"
    score: int  # 0-100
    evidence_paths: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvalResult:
    """Per-run evaluation results following the eval.json contract."""
    schema_version: str
    run_id: str
    use_case_id: Optional[str]  # e.g., "asr_smoke_v1", "meeting_analysis"
    model_id: Optional[str]  # e.g., "faster_whisper_large_v3"
    params: Dict[str, Any]  # Run parameters
    metrics: Dict[str, float]  # e.g., {"wer": 0.12, "der": 0.08}
    checks: List[Check]
    findings: List[Finding]
    generated_at: str  # ISO8601
    score_cards: List[ScoreCard] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "use_case_id": self.use_case_id,
            "model_id": self.model_id,
            "params": self.params,
            "metrics": self.metrics,
            "checks": [c.to_dict() for c in self.checks],
            "findings": [f.to_dict() for f in self.findings],
            "score_cards": [s.to_dict() for s in self.score_cards],
            "generated_at": self.generated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalResult":
        """Load from dict, handling nested objects."""
        return cls(
            schema_version=data["schema_version"],
            run_id=data["run_id"],
            use_case_id=data.get("use_case_id"),
            model_id=data.get("model_id"),
            params=data.get("params", {}),
            metrics=data.get("metrics", {}),
            checks=[Check(**c) for c in data.get("checks", [])],
            findings=[Finding(**f) for f in data.get("findings", [])],
            score_cards=[ScoreCard(**s) for s in data.get("score_cards", [])],
            generated_at=data["generated_at"]
        )


def create_eval_stub(run_id: str, message: str = "No evaluation performed") -> EvalResult:
    """Create a minimal eval result for runs without evaluation."""
    return EvalResult(
        schema_version="1",
        run_id=run_id,
        use_case_id=None,
        model_id=None,
        params={},
        metrics={},
        checks=[Check(
            name="eval_availability",
            passed=False,
            severity="info",
            message=message,
            evidence_paths=[]
        )],
        findings=[],
        generated_at=datetime.now().isoformat()
    )
