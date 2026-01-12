#!/usr/bin/env python3
"""
Arsenal Documentation Generator.

Merges:
  - Registry metadata (discovery + wiring)
  - Model configs (declared facts)
  - Runs summaries (observed facts)

Outputs:
  - docs/arsenal.json (machine-readable)
  - docs/ARSENAL.md (human-readable)

Usage:
  uv run python scripts/generate_arsenal.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add harness to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.registry import ModelRegistry
from harness.model_card import ModelCard, DeclaredCapability, EvidenceEntry, OfficialSource, EvaluationInfo, DeploymentInfo, HardwareInfo, Artifact, InstallInfo, ObservedMetrics, ClaimsInfo
from harness.taxonomy import TaskType, TaskRole, EvidenceGrade


def get_git_info() -> dict:
    """Get git info for deterministic generation and freshness tracking."""
    import subprocess
    info = {"commit": "unknown", "tree": "unknown", "dirty": False}
    
    try:
        # Commit hash (for traceability)
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True
        )
        info["commit"] = result.stdout.strip()
        
        # Tree hash (changes when tracked files change - real freshness indicator)
        result = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            capture_output=True, text=True, check=True
        )
        info["tree"] = result.stdout.strip()[:12]
        
        # Check if dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True
        )
        info["dirty"] = len(result.stdout.strip()) > 0
        
    except Exception:
        pass
    
    return info


def load_model_config(model_id: str) -> Dict[str, Any]:
    """Load model config from models/<model_id>/config.yaml."""
    config_path = Path(f"models/{model_id}/config.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_all_task_evidence(model_id: str) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Load best run data for a model across all tasks.
    
    Scans runs/<model_id>/*/*.json for all task directories.
    Returns dict mapping task -> best evidence for that task.
    """
    runs_base = Path(f"runs/{model_id}")
    if not runs_base.exists():
        return {}
    
    task_evidence = {}
    
    # Scan all task subdirectories
    for task_dir in runs_base.iterdir():
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        evidence = load_best_run_for_task(model_id, task)
        if evidence:
            task_evidence[task] = evidence
    
    return task_evidence


    return task_evidence


def load_best_run_for_task(model_id: str, task: str) -> Optional[Dict[str, Any]]:
    """
    Load best run data for a specific task.
    
    Deterministic selection rule:
    1. Valid runs only (unless no valid runs exist)
    2. Preferred dataset (from model config 'evaluation.default_datasets')
    3. Grade: GOLDEN_BATCH > SMOKE > ADHOC
    4. Newest verified_at date
    """
    runs_dir = Path(f"runs/{model_id}/{task}")
    if not runs_dir.exists():
        return None
    
    # Find all jSON run files
    run_files = sorted([
        f for f in runs_dir.glob("*.json") 
        if f.name != "summary.json" and not f.name.startswith(".")
    ])
    
    if not run_files:
        return None
        
    runs_data = []
    for f in run_files:
        try:
            with open(f) as rf:
                d = json.load(rf)
                d["_filename"] = f.name
                runs_data.append(d)
        except Exception:
            continue
            
    if not runs_data:
        return None

    # Helper to check validity
    def is_valid(r):
        # Check explicit valid flag first, then infer from gates
        if "valid" in r: return r["valid"]
        # Legacy inference
        gates = r.get("evidence", {}).get("sanity_gates", {})
        return r.get("evidence", {}).get("wer_valid", True) and not r.get("gates", {}).get("has_failure", False)

    # Helper to get grade logic
    def grade_score(r):
        g = r.get("evidence", {}).get("grade", "adhoc")
        if "golden" in g: return 3
        if "smoke" in g: return 2
        return 1
        
    # Helper to get timestamp
    def get_ts(r):
        return r.get("manifest", {}).get("timestamp", "")
        
    # Sort: Valid > Grade > Timestamp
    runs_data.sort(key=lambda r: (
        is_valid(r),
        grade_score(r),
        get_ts(r)
    ), reverse=True)
    
    # Pick winner
    run_data = runs_data[0]
    
    # --- Normalization to EvidenceEntry dict format ---
    metrics = run_data.get("metrics", {})
    system = run_data.get("system", {})
    manifest = run_data.get("manifest", {})
    output = run_data.get("output", {})
    evidence = run_data.get("evidence", {})
    gates = run_data.get("gates", {})
    
    # Determine grade enum
    grade_str = evidence.get("grade", "adhoc")
    if "golden" in grade_str: grade_enum = EvidenceGrade.GOLDEN_BATCH
    elif "smoke" in grade_str: grade_enum = EvidenceGrade.SMOKE
    else: grade_enum = EvidenceGrade.ADHOC
    
    result = {
        "task": task,
        "dataset_id": evidence.get("dataset_id") or run_data.get("input", {}).get("dataset_id", "unknown"),
        "evidence_grade": grade_enum,
        "device": system.get("device", "unknown"),
        "verified_at": manifest.get("timestamp", "").split("T")[0],
        "metrics": {},
        "gates": {},
        "valid": is_valid(run_data),
        "invalid_reasons": []
    }
    
    # If invalid, try to find reasons
    if not result["valid"]:
        if run_data.get("output_quality", {}).get("has_failure"):
            result["invalid_reasons"].append("output_quality_failure")
        if not evidence.get("wer_valid", True):
             result["invalid_reasons"].append("wer_invalid")
        result["invalid_reasons"].extend(run_data.get("evidence", {}).get("invalid_reasons", []))

    # Task specific extraction (Keep existing logic but streamlined)
    if task == 'asr':
        wer = metrics.get("wer_mean") or metrics.get("wer")
        rtf = metrics.get("rtf_median") or metrics.get("rtf")
        latency = metrics.get("latency_ms_p50") or metrics.get("latency_ms")
        
        result["metrics"] = {
            "wer": wer,
            "rtf": rtf,
            "latency_ms": latency
        }
        result["gates"] = {
            "wer_valid": evidence.get("wer_valid", True),
            "is_truncated": run_data.get("output_quality", {}).get("is_truncated")
        }
        
    elif task == 'tts':
        result["metrics"] = {
            "latency_ms": metrics.get("latency_ms_avg") or metrics.get("latency_ms"),
            "rtf": metrics.get("rtf")
        }
        result["gates"] = {
            # Propagate raw gates
            **gates
        }
        
    elif task == 'vad':
         result["metrics"] = {
            "speech_ratio": metrics.get("speech_ratio"),
            "num_segments": metrics.get("num_segments"),
            "rtf": metrics.get("rtf")
        }
         result["gates"] = gates

    elif task == 'diarization':
        result["metrics"] = {
            "num_speakers": metrics.get("num_speakers_pred"),
            "rtf": metrics.get("rtf")
        }
        result["gates"] = gates
        
    elif task == 'v2v':
        result["metrics"] = {
            "latency_ms": metrics.get("latency_ms"),
            "turn_latency_ms": metrics.get("turn_latency_ms")
        }
        result["gates"] = gates
        
    else:
        result["metrics"] = metrics
        
    return result


def build_all_cards() -> List[ModelCard]:
    """Build ModelCard for all registered models."""
    cards = []
    
    for model_id in ModelRegistry.list_models():
        registry_meta = ModelRegistry.get_model_metadata(model_id) or {}
        config = load_model_config(model_id)
        
        # Load evidence for all tasks
        task_evidence = load_all_task_evidence(model_id)
        
        # For backward compatibility, use ASR evidence as the primary runs_summary
        runs_summary = task_evidence.get('asr')
        
        # Store all task evidence
        evidence_data = list(task_evidence.values())
        
        card = ModelCard.from_sources(model_id, registry_meta, config, runs_summary, evidence_data)
        cards.append(card)
    
    return cards


# Backward compatibility alias
def load_runs_summary(model_id: str) -> Optional[Dict[str, Any]]:
    """Backward compatibility: load ASR evidence only."""
    task_evidence = load_all_task_evidence(model_id)
    return task_evidence.get('asr')


def generate_json(cards: List[ModelCard], output_path: Path):
    """Generate arsenal.json (deterministic - no timestamps)."""
    # Sort cards for deterministic output
    sorted_cards = sorted(cards, key=lambda c: c.model_id)
    git_info = get_git_info()
    
    data = {
        "arsenal_schema_version": 1,
        "generated_from_commit": git_info["commit"],
        "generated_from_tree": git_info["tree"],  # Real freshness indicator
        "generator_version": "1.0",
        "models_count": len(sorted_cards),
        "models": [card.to_dict() for card in sorted_cards]
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Generated {output_path}")


def generate_markdown(cards: List[ModelCard], output_path: Path):
    """Generate ARSENAL.md."""
    lines = []
    
    # Header (deterministic - no timestamps)
    git_info = get_git_info()
    lines.append("# Model Arsenal")
    lines.append("")
    lines.append(f"> Auto-generated from registry + configs + runs. Do not edit manually.")
    lines.append(f"> Generated from commit: {git_info['commit']} (tree: {git_info['tree']})")
    if git_info['dirty']:
        lines.append(f"> ⚠️ Working directory has uncommitted changes")
    lines.append("")
    
    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Model | Status | Capabilities | Runtimes | Targets | WER | RTF |")
    lines.append("|-------|--------|--------------|----------|---------|-----|-----|")
    
    for card in sorted(cards, key=lambda c: (c.status != "production", c.model_id)):
        caps = ", ".join(card.capabilities) if card.capabilities else "-"
        runtimes = ", ".join(card.deployment.runtimes) if card.deployment.runtimes else "-"
        targets = ", ".join(card.deployment.targets) if card.deployment.targets else "-"
        wer = f"{card.observed.wer_mean:.1%}" if card.observed.wer_mean else "-"
        rtf = f"{card.observed.rtf_median:.2f}x" if card.observed.rtf_median else "-"
        
        status_emoji = {"production": "✅", "candidate": "🟡", "experimental": "🔬", "deprecated": "⚠️"}.get(card.status, "❓")
        
        lines.append(f"| **{card.model_id}** | {status_emoji} {card.status} | {caps} | {runtimes} | {targets} | {wer} | {rtf} |")
    
    lines.append("")
    
    # Per-model sections
    lines.append("---")
    lines.append("")
    lines.append("## Model Details")
    lines.append("")
    
    for card in sorted(cards, key=lambda c: c.model_id):
        lines.append(f"### {card.model_id}")
        lines.append("")
        lines.append(f"**{card.model_name}** by {card.provider or 'Unknown'}")
        lines.append("")
        
        if card.description:
            lines.append(f"> {card.description.strip().split(chr(10))[0]}")
            lines.append("")
        
        # Quick facts
        lines.append("| Attribute | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| Status | {card.status} |")
        lines.append(f"| Capabilities | {', '.join(card.capabilities) or '-'} |")
        lines.append(f"| Accelerators | {', '.join(card.hardware.accelerators_supported)} |")
        lines.append(f"| Offline | {'✅' if card.deployment.offline_capable else '❌'} |")
        lines.append(f"| License | {card.license or 'Unknown'} |")
        lines.append("")
        
        # When to use
        if card.best_app_types:
            lines.append(f"**Best for:** {', '.join(card.best_app_types)}")
            lines.append("")
        if card.poor_app_types:
            lines.append(f"**Avoid if:** {', '.join(card.poor_app_types)}")
            lines.append("")
        
        # How to run
        lines.append("**Run:**")
        lines.append(f"```bash")
        lines.append(f"make asr MODEL={card.model_id} DATASET=primary")
        lines.append(f"```")
        lines.append("")
        
        # Official Sources
        if card.official_sources:
            lines.append("**Sources:**")
            for s in card.official_sources:
                url_str = f"[{s.kind}]({s.url})" if s.url else s.kind
                note_str = f" - {s.note}" if s.note else ""
                lines.append(f"- {url_str}{note_str}")
            lines.append("")

        # Declared Capabilities
        if card.declared_capabilities:
            lines.append("#### Declared Capabilities (Intent)")
            lines.append("| Task | Role | Confidence | Notes |")
            lines.append("|------|------|------------|-------|")
            for cap in card.declared_capabilities:
                lines.append(f"| {cap.task} | {cap.role} | {cap.confidence} | {cap.notes} |")
            lines.append("")
            
        # Claims
        if card.claims.claimed_strengths:
            lines.append(f"**Strengths:** {', '.join(card.claims.claimed_strengths)}")
            lines.append("")

        # Observed Evidence
        if card.evidence:
            lines.append("#### Observed Evidence")
            lines.append("| Task | Grade | Device | Metrics | Gates | Ver. |")
            lines.append("|------|-------|--------|---------|-------|------|")
            for ev in sorted(card.evidence, key=lambda e: e.task):
                # Metrics Formatting
                m_parts = []
                for k, v in ev.metrics.items():
                    if v is None: continue
                    if isinstance(v, float):
                        if "rtf" in k: val = f"{v:.2f}x"
                        elif "time" in k or "latency" in k: val = f"{v:.0f}ms"
                        elif "wer" in k or "cer" in k: val = f"{v:.1%}"
                        else: val = f"{v:.2f}"
                    else:
                        val = str(v)
                    m_parts.append(f"{k}:{val}")
                metrics_str = "<br>".join(m_parts) if m_parts else "-"
                
                # Gates Formatting
                failures = [k for k, v in ev.gates.items() if (k == "has_failure" and v) or (k != "has_failure" and v is False)]
                if failures:
                    gates_str = "❌ " + ", ".join(failures)
                else:
                    gates_str = "✅ Pass"

                lines.append(f"| {ev.task} | {ev.evidence_grade} | {ev.device or '-'} | {metrics_str} | {gates_str} | {ev.verified_at or '-'} |")
            lines.append("")
        
        # Known issues
        if card.install.known_issues:
            lines.append("**Known Issues:**")
            for issue in card.install.known_issues:
                lines.append(f"- {issue}")
            lines.append("")
        
        if card.hardware.notes:
            lines.append("**Hardware Notes:**")
            for note in card.hardware.notes:
                lines.append(f"- {note}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"✓ Generated {output_path}")


def generate_use_cases(cards: List[ModelCard], output_path: Path):
    """Generate USE_CASES.md - model recommendations per use-case."""
    use_cases_file = Path("data/use_cases.yaml")
    if not use_cases_file.exists():
        print(f"⚠️  Skipping USE_CASES.md - {use_cases_file} not found")
        return
    
    with open(use_cases_file) as f:
        use_cases = yaml.safe_load(f)
    
    lines = [
        "# Use Case Recommendations",
        "",
        "> Auto-generated from `data/use_cases.yaml` + model evidence.",
        "> Do not edit manually.",
        "",
    ]
    
    for use_case in use_cases:
        lines.append(f"## {use_case['title']}")
        lines.append("")
        lines.append(f"_{use_case.get('description', '')}_")
        lines.append("")
        
        requires = use_case.get('requires', {})
        prefers = use_case.get('prefers', {})
        reject_if = use_case.get('reject_if', {})
        
        # Score and rank models for this use case
        recommendations = []
        rejections = []
        
        for card in cards:
            reasons_reject = []
            reasons_recommend = []
            score = 0
            
            # Check required capability
            req_cap = requires.get('capability')
            req_caps_any = requires.get('capability_any', [])
            if req_cap and req_cap not in card.capabilities:
                reasons_reject.append(f"missing capability: {req_cap}")
            if req_caps_any and not any(c in card.capabilities for c in req_caps_any):
                reasons_reject.append(f"missing capabilities: {req_caps_any}")
            
            # Check required targets
            req_targets = requires.get('targets_any', [])
            if req_targets:
                card_targets = card.deployment.targets if card.deployment else []
                if not any(t in card_targets for t in req_targets):
                    reasons_reject.append(f"wrong targets: {card_targets}")
            
            # Check evidence validity
            if reject_if.get('evidence_invalid') and not card.observed.accelerators_verified:
                reasons_reject.append("no verified evidence")
            
            # Check WER threshold
            wer_max = reject_if.get('wer_above') or prefers.get('wer_max')
            if wer_max and card.observed.wer_mean and card.observed.wer_mean > wer_max:
                reasons_reject.append(f"WER {card.observed.wer_mean:.1%} > {wer_max:.0%}")
            
            # Scoring
            if card.observed.wer_mean:
                score += (1 - card.observed.wer_mean) * prefers.get('wer_weight', 0.5) * 100
                reasons_recommend.append(f"WER={card.observed.wer_mean:.1%}")
            if card.observed.rtf_median:
                score += (1 - min(card.observed.rtf_median, 1)) * prefers.get('rtf_weight', 0.3) * 100
                reasons_recommend.append(f"RTF={card.observed.rtf_median:.2f}x")
            
            if reasons_reject:
                rejections.append((card.model_id, reasons_reject))
            else:
                recommendations.append((card.model_id, score, reasons_recommend))
        
        # Sort by score descending
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        if recommendations:
            lines.append("### ✅ Recommended Models")
            lines.append("")
            lines.append("| Model | Score | Reasons |")
            lines.append("|-------|-------|---------|")
            for model_id, score, reasons in recommendations[:5]:  # Top 5
                reasons_str = ", ".join(reasons) if reasons else "-"
                lines.append(f"| **{model_id}** | {score:.0f} | {reasons_str} |")
            lines.append("")
        
        if rejections:
            lines.append("### ❌ Not Recommended")
            lines.append("")
            for model_id, reasons in rejections:
                lines.append(f"- **{model_id}**: {', '.join(reasons)}")
            lines.append("")
        
        if use_case.get('notes'):
            lines.append(f"> **Note:** {use_case['notes']}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"✓ Generated {output_path}")


def main():
    print("=== Arsenal Documentation Generator ===")
    print()
    
    # Build all cards
    cards = build_all_cards()
    print(f"Found {len(cards)} registered models")
    
    # Create output directory
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Generate outputs
    generate_json(cards, docs_dir / "arsenal.json")
    generate_markdown(cards, docs_dir / "ARSENAL.md")
    generate_use_cases(cards, docs_dir / "USE_CASES.md")
    
    print()
    print("🎉 Arsenal docs regenerated!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
