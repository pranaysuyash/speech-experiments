"""
ModelCard Schema - Merged view of model metadata for Arsenal doc generation.

Sources merged:
1. Registry metadata (discovery + wiring)
2. Model config.yaml (declared facts)
3. Runs (observed facts)

Registry stays minimal. Config owns doc-like fields. Runs own evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from pathlib import Path

from .taxonomy import TaskType, TaskRole, EvidenceGrade


@dataclass
class OfficialSource:
    """Official documentation link."""
    kind: str  # "hf", "paper", "repo", "web"
    url: Optional[str] = None
    note: str = ""


@dataclass
class DeclaredCapability:
    """A capability explicitly declared by the model."""
    task: TaskType
    role: TaskRole
    confidence: str = "unknown"
    notes: Optional[str] = None
    source: str = "config"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task.value,
            "role": self.role.value,
            "confidence": self.confidence,
            "notes": self.notes,
            "source": self.source
        }

@dataclass
class ClaimsInfo:
    """Narrative claims about the model."""
    claimed_strengths: List[str] = field(default_factory=list)
    claimed_limitations: List[str] = field(default_factory=list)
    recommended_usage: List[str] = field(default_factory=list)


@dataclass
class EvaluationInfo:
    """Testing priorities and dataset overrides."""
    primary_tasks: List[str] = field(default_factory=list)
    secondary_tasks: List[str] = field(default_factory=list)
    skip_tasks: List[str] = field(default_factory=list)
    default_datasets: Dict[str, str] = field(default_factory=dict)

@dataclass
class DeploymentInfo:
    """Deployment scope and targets."""
    runtimes: List[str] = field(default_factory=list)  # ["local", "api", "cli", "browser"]
    offline_capable: bool = False
    targets: List[str] = field(default_factory=list)  # ["desktop", "server", "mobile", "web"]
    notes: str = ""


@dataclass
class HardwareInfo:
    """Hardware requirements and constraints."""
    accelerators_supported: List[str] = field(default_factory=lambda: ["cpu"])
    min_vram_gb: Optional[float] = None
    ram_gb_recommended: Optional[float] = None
    disk_gb_model: Optional[float] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class Artifact:
    """Packaging artifact (model format/quantization)."""
    type: str  # "pytorch", "gguf", "onnx", "coreml"
    precision: List[str] = field(default_factory=list)  # ["fp16", "bf16", "q4", "q8"]
    notes: str = ""


@dataclass
class InstallInfo:
    """Installation and dependency information."""
    method: str = "pip"  # "pip", "uv", "docker", "binary", "cli"
    deps_pain: str = "low"  # "low", "medium", "high"
    known_issues: List[str] = field(default_factory=list)


@dataclass
class EvidenceEntry:
    """A single unit of observed evidence."""
    task: TaskType  # e.g. "asr"
    dataset_id: str  # e.g. "librispeech_test_clean"
    evidence_grade: EvidenceGrade  # e.g. "golden_batch"
    metrics: Dict[str, Any]  # e.g. {"wer": 0.05}
    valid: bool  # derived from gates
    invalid_reasons: List[str]
    device: Optional[str] = None  # e.g. "mps"
    run_date: Optional[str] = None
    verified_at: Optional[str] = None  # ISO date
    gates: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        from enum import Enum # Import locally to avoid circular dependency if Taxonomy imports ModelCard
        return {
            "task": self.task.value if isinstance(self.task, Enum) else self.task,
            "dataset_id": self.dataset_id,
            "evidence_grade": self.evidence_grade.value if isinstance(self.evidence_grade, Enum) else self.evidence_grade,
            "metrics": self.metrics,
            "gates": self.gates,
            "valid": self.valid,
            "invalid_reasons": self.invalid_reasons,
            "device": self.device,
            "run_date": self.run_date,
            "verified_at": self.verified_at
        }


@dataclass
class ObservedMetrics:
    """Metrics derived from runs (auto-filled). DEPRECATED: Use EvidenceEntry list instead."""
    wer_mean: Optional[float] = None
    cer_mean: Optional[float] = None
    rtf_median: Optional[float] = None
    latency_ms_p50: Optional[float] = None
    failure_rate: Optional[float] = None
    failure_modes: List[str] = field(default_factory=list)
    accelerators_verified: List[str] = field(default_factory=list)
    golden_set_version: Optional[str] = None
    last_verified_commit: Optional[str] = None
    last_verified_at: Optional[str] = None


@dataclass
class ModelCard:
    """
    Complete model card - merged from registry + config + runs.
    
    Required fields (must exist):
        model_id, capabilities, status
    
    Optional fields (can be missing, generator handles gracefully):
        Everything else
    """
    # Identity (from registry + config)
    model_id: str
    model_name: str = ""
    provider: str = ""
    version: str = ""
    license: str = ""
    description: str = ""
    
    # Official Sources (from config)
    official_sources: List[OfficialSource] = field(default_factory=list)
    
    # Claims (from config)
    claims: ClaimsInfo = field(default_factory=ClaimsInfo)
    
    # Declared Intent (derived from config)
    declared_capabilities: List[DeclaredCapability] = field(default_factory=list)
    
    # Evaluation Config (from config)
    evaluation: EvaluationInfo = field(default_factory=EvaluationInfo)
    
    # Status (from registry)
    status: str = "experimental"  # experimental, candidate, production, deprecated
    capabilities: List[str] = field(default_factory=list)  # asr, tts, chat, mt
    modes: List[str] = field(default_factory=list)  # batch, streaming, cli
    hash: str = ""
    
    # Deployment (from config)
    deployment: DeploymentInfo = field(default_factory=DeploymentInfo)
    
    # Hardware (from config)
    hardware: HardwareInfo = field(default_factory=HardwareInfo)
    
    # Artifacts (from config)
    artifacts: List[Artifact] = field(default_factory=list)
    
    # Install (from config)
    install: InstallInfo = field(default_factory=InstallInfo)
    
    # Observed (from runs - auto-filled)
    observed: ObservedMetrics = field(default_factory=ObservedMetrics)
    
    # Evidence (multi-capability)
    evidence: List[EvidenceEntry] = field(default_factory=list)
    
    # Recommendation (from config)
    best_app_types: List[str] = field(default_factory=list)
    poor_app_types: List[str] = field(default_factory=list)
    interaction_style: str = "batch"  # batch, near_real_time, streaming, conversational
    
    # Links (from config)
    paper_url: Optional[str] = None
    repo_url: Optional[str] = None
    
    @classmethod
    def from_sources(cls, 
                     model_id: str,
                     registry_meta: Dict[str, Any],
                     config: Dict[str, Any],
                     runs_summary: Optional[Dict[str, Any]] = None,
                     evidence_data: Optional[List[Dict[str, Any]]] = None) -> "ModelCard":
        """
        Merge registry + config + runs into a ModelCard.
        
        Args:
            model_id: Model identifier
            registry_meta: From ModelRegistry.get_model_metadata()
            config: From models/<model_id>/config.yaml
            runs_summary: Latest run summary for the model (legacy ASR)
            evidence_data: List of normalized evidence entries for all tasks
        """
        runs_summary = runs_summary or {}
        evidence_data = evidence_data or []
        
        # Parse nested config sections
        config_meta = config.get("metadata", {})
        config_deploy = config.get("deployment", {})
        config_hw = config.get("hardware", {})
        config_install = config.get("install", {})
        config_recommend = config.get("recommendation", {})
        
        config_sources = config.get("official_sources", [])
        config_eval = config.get("evaluation", {})
        
        # Build new sections
        official_sources = [OfficialSource(**s) for s in config_sources]
        evaluation = EvaluationInfo(**config_eval)
        
        # Build deployment info
        deployment = DeploymentInfo(
            runtimes=config_deploy.get("runtimes", ["local"]),
            offline_capable=config_deploy.get("offline_capable", True),
            targets=config_deploy.get("targets", ["desktop"]),
            notes=config_deploy.get("notes", "")
        )
        
        # Build hardware info - merge registry accelerators with config details
        hardware = HardwareInfo(
            accelerators_supported=registry_meta.get("hardware", config_hw.get("accelerators_supported", ["cpu"])),
            min_vram_gb=config_hw.get("min_vram_gb"),
            ram_gb_recommended=config_hw.get("ram_gb_recommended"),
            disk_gb_model=config_hw.get("disk_gb_model"),
            notes=config_hw.get("notes", [])
        )
        
        # Parse claims
        claims_data = config.get("claims", {})
        claims_info = ClaimsInfo(
            claimed_strengths=claims_data.get("claimed_strengths", []),
            claimed_limitations=claims_data.get("claimed_limitations", []),
            recommended_usage=claims_data.get("recommended_usage", [])
        )
        
        # Parse declared capabilities
        declared = []
        for d in config.get("declared_capabilities", []):
            declared.append(DeclaredCapability(
                task=TaskType(d["task"]),
                role=TaskRole(d["role"]),
                confidence=d.get("confidence", "unknown"),
                notes=d.get("notes"),
                source=d.get("source", "config")
            ))
        
        # FALLBACK: If no declared_capabilities in config, infer from registry
        if not declared and registry_meta.get("capabilities"):
            for cap in registry_meta["capabilities"]:
                try:
                    declared.append(DeclaredCapability(
                        task=TaskType(cap),
                        role=TaskRole.PRIMARY,  # Assume primary if not specified
                        confidence="inferred",
                        notes="Inferred from registry capabilities",
                        source="registry"
                    ))
                except ValueError:
                    pass  # Skip unknown task types
        
        # Build artifacts
        artifacts = []
        for art in config.get("artifacts", []):
            artifacts.append(Artifact(
                type=art.get("type", "pytorch"),
                precision=art.get("precision", []),
                notes=art.get("notes", "")
            ))
        
        # Build install info
        install = InstallInfo(
            method=config_install.get("method", "pip"),
            deps_pain=config_install.get("deps_pain", "low"),
            known_issues=config_install.get("known_issues", [])
        )
        
        # Build observed metrics from runs
        observed = ObservedMetrics(
            wer_mean=runs_summary.get("wer"),
            cer_mean=runs_summary.get("cer"),
            rtf_median=runs_summary.get("rtf"),
            latency_ms_p50=runs_summary.get("latency_ms"),
            failure_rate=runs_summary.get("failure_rate"),
            failure_modes=runs_summary.get("failure_modes", []),
            accelerators_verified=runs_summary.get("accelerators_verified", []),
            golden_set_version=runs_summary.get("golden_set_version"),
            last_verified_commit=runs_summary.get("commit"),
            last_verified_at=runs_summary.get("last_run")
        )
        
        # Build evidence list
        evidence_list = []
        if evidence_data:
            for ev in evidence_data:
                try:
                    evidence_list.append(EvidenceEntry(**ev))
                except Exception as e:
                    print(f"Warning: Failed to create EvidenceEntry for {model_id}: {e}")
        
        return cls(
            model_id=model_id,
            model_name=config.get("model_name", model_id),
            provider=config_meta.get("provider", ""),
            version=registry_meta.get("version", config_meta.get("version", "")),
            license=config_meta.get("license", ""),
            description=config_meta.get("description", ""),
            official_sources=official_sources,
            claims=claims_info,
            declared_capabilities=declared,
            evaluation=evaluation,
            status=registry_meta.get("status", "experimental"),
            capabilities=registry_meta.get("capabilities", []),
            modes=registry_meta.get("modes", []),
            hash=registry_meta.get("hash", ""),
            deployment=deployment,
            hardware=hardware,
            artifacts=artifacts,
            install=install,
            observed=observed,
            evidence=evidence_list,
            best_app_types=config_recommend.get("best_app_types", []),
            poor_app_types=config_recommend.get("poor_app_types", []),
            interaction_style=config_recommend.get("interaction_style", "batch"),
            paper_url=config_meta.get("paper_url"),
            repo_url=config_meta.get("repo_url")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "model_id": self.model_id,
            "evidence": [asdict(e) for e in self.evidence],
            "model_name": self.model_name,
            "provider": self.provider,
            "version": self.version,
            "license": self.license,
            "description": self.description,
            "official_sources": [asdict(s) for s in self.official_sources],
            "claims": asdict(self.claims),
            "declared_capabilities": [asdict(d) for d in self.declared_capabilities],
            "evaluation": asdict(self.evaluation),
            "status": self.status,
            "capabilities": self.capabilities,
            "modes": self.modes,
            "hash": self.hash,
            "deployment": {
                "runtimes": self.deployment.runtimes,
                "offline_capable": self.deployment.offline_capable,
                "targets": self.deployment.targets,
                "notes": self.deployment.notes
            },
            "hardware": {
                "accelerators_supported": self.hardware.accelerators_supported,
                "min_vram_gb": self.hardware.min_vram_gb,
                "ram_gb_recommended": self.hardware.ram_gb_recommended,
                "disk_gb_model": self.hardware.disk_gb_model,
                "notes": self.hardware.notes
            },
            "artifacts": [{"type": a.type, "precision": a.precision, "notes": a.notes} for a in self.artifacts],
            "install": {
                "method": self.install.method,
                "deps_pain": self.install.deps_pain,
                "known_issues": self.install.known_issues
            },
            "observed": {
                "wer_mean": self.observed.wer_mean,
                "cer_mean": self.observed.cer_mean,
                "rtf_median": self.observed.rtf_median,
                "latency_ms_p50": self.observed.latency_ms_p50,
                "failure_rate": self.observed.failure_rate,
                "failure_modes": self.observed.failure_modes,
                "accelerators_verified": self.observed.accelerators_verified,
                "golden_set_version": self.observed.golden_set_version,
                "last_verified_commit": self.observed.last_verified_commit,
                "last_verified_at": self.observed.last_verified_at
            },
            "recommendation": {
                "best_app_types": self.best_app_types,
                "poor_app_types": self.poor_app_types,
                "interaction_style": self.interaction_style
            },
            "links": {
                "paper_url": self.paper_url,
                "repo_url": self.repo_url
            }
        }


# Required fields for promotion gates
REQUIRED_FOR_CANDIDATE = ["capabilities", "deployment", "hardware"]
REQUIRED_FOR_PRODUCTION = REQUIRED_FOR_CANDIDATE + ["observed"]


def validate_for_promotion(card: ModelCard, target_status: str) -> tuple[bool, List[str]]:
    """
    Validate if a model card meets promotion requirements.
    
    Experimental: minimal requirements (just capabilities)
    Candidate: identity + deployment complete, at least one run
    Production: evidence-backed (verified accelerators, quality metrics, recommendations)
    
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    
    # All statuses require capabilities
    if not card.capabilities:
        issues.append("Missing capabilities")
    
    # === CANDIDATE requires: ===
    if target_status in ("candidate", "production"):
        # Identity fields must be filled
        if not card.provider:
            issues.append("Missing provider (required for candidate)")
        if not card.license:
            issues.append("Missing license (required for candidate)")
        if not card.description:
            issues.append("Missing description (required for candidate)")
        
        # Deployment must be specified
        if not card.deployment.runtimes:
            issues.append("Missing deployment.runtimes")
        if not card.deployment.targets:
            issues.append("Missing deployment.targets")
        
        # Hardware must be specified
        if not card.hardware.accelerators_supported:
            issues.append("Missing hardware.accelerators_supported")
    
    # === PRODUCTION requires (above plus): ===
    if target_status == "production":
        # Must have at least one verified accelerator (evidence of a successful run)
        if not card.observed.accelerators_verified:
            issues.append("No accelerators verified - requires at least one successful run")
        
        # ASR models must have WER observation
        if "asr" in card.capabilities and card.observed.wer_mean is None:
            issues.append("ASR model missing WER observation")
        
        # Must have recommendation guidance
        if not card.best_app_types:
            issues.append("Missing recommendation.best_app_types")
        
        # Must have a link (repo or paper)
        if not card.repo_url and not card.paper_url:
            issues.append("Missing links (repo_url or paper_url required)")
        
        # Failure rate must be acceptable
        if card.observed.failure_rate and card.observed.failure_rate > 0.1:
            issues.append(f"High failure rate: {card.observed.failure_rate:.1%}")
        
        # Must have a last_verified timestamp
        if not card.observed.last_verified_at:
            issues.append("No last_verified_at - never been run")
    
    return len(issues) == 0, issues


def check_current_status_valid(card: ModelCard) -> tuple[bool, List[str], str]:
    """
    Check if model's current status is backed by evidence.
    
    Returns:
        (is_valid, issues, suggested_status)
    """
    is_valid, issues = validate_for_promotion(card, card.status)
    
    if is_valid:
        return True, [], card.status
    
    # Suggest downgrade if current status isn't earned
    if card.status == "production":
        prod_valid, _ = validate_for_promotion(card, "candidate")
        if prod_valid:
            return False, issues, "candidate"
        return False, issues, "experimental"
    elif card.status == "candidate":
        return False, issues, "experimental"
    
    return False, issues, "experimental"

