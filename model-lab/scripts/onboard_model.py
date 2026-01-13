#!/usr/bin/env python3
"""
Model Onboarding Script.

Validates a model config, runs required smoke tests for declared capabilities,
and generates valid evidence artifacts. This is the canonical way to add
a new model to the arsenal.

Exit Codes:
    0 - Success, all tests passed
    1 - Config validation failed
    2 - Missing required environment variable (e.g., HF_TOKEN)
    3 - Some smoke tests failed
    4 - All smoke tests failed

Usage:
    python scripts/onboard_model.py --model <model_id>
    python scripts/onboard_model.py --model <model_id> --tasks asr,vad  # Specific tasks
    python scripts/onboard_model.py --model <model_id> --golden  # Also run golden tests
    python scripts/onboard_model.py --model <model_id> --dry-run  # Validate only
"""

import argparse
import os
import subprocess
import sys
import yaml
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.registry import ModelRegistry

# Exit codes
EXIT_SUCCESS = 0
EXIT_CONFIG_INVALID = 1
EXIT_MISSING_ENV = 2
EXIT_SOME_FAILED = 3
EXIT_ALL_FAILED = 4


# Models that require specific environment variables
ENV_REQUIREMENTS = {
    "pyannote_diarization": ["HF_TOKEN"],
    "lfm2_5_audio": [],  # No special env needed
    "whisper": [],
    "faster_whisper": [],
    "seamlessm4t": [],
    "distil_whisper": [],
    "heuristic_diarization": [],
    "silero_vad": [],
    "whisper_cpp": [],  # Requires binary, not env var
}


def validate_model_config(model_id: str) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    """Validate that model config loads correctly."""
    errors = []
    meta = None
    
    try:
        registry = ModelRegistry()
        meta = registry.get_model_metadata(model_id)
        
        if meta is None:
            errors.append(f"Model {model_id} not found in registry")
            return False, errors, None
        
        # Check for capabilities
        if not meta.get("capabilities"):
            errors.append("No capabilities found in registry metadata")
        
    except Exception as e:
        errors.append(f"Failed to load config: {e}")
        return False, errors, None
    
    return len(errors) == 0, errors, meta


def check_env_requirements(model_id: str) -> Tuple[bool, List[str]]:
    """Check if required environment variables are set."""
    required = ENV_REQUIREMENTS.get(model_id, [])
    missing = [var for var in required if not os.environ.get(var)]
    
    if missing:
        return False, [f"Missing required env var: {var}" for var in missing]
    return True, []


def get_smoke_datasets(task: str) -> List[str]:
    """Get available smoke datasets for a task."""
    task_datasets = {
        "asr": ["asr_smoke_v1"],
        "vad": ["vad_smoke_v1"],
        "diarization": ["diar_smoke_v1"],
        "v2v": ["v2v_smoke_v1"],
        "tts": ["tts_smoke_v1"],
    }
    return task_datasets.get(task, [])


def get_golden_datasets(task: str) -> List[str]:
    """Get available golden datasets for a task."""
    task_datasets = {
        "asr": ["primary"],
        "diarization": [],
        "vad": [],
        "v2v": [],
    }
    return task_datasets.get(task, [])


def run_smoke_test(model_id: str, task: str, dataset: str, dry_run: bool = False) -> Tuple[bool, str]:
    """Run a single smoke test for a model/task/dataset."""
    script_map = {
        "asr": "scripts/run_asr.py",
        "vad": "scripts/run_vad.py",
        "diarization": "scripts/run_diarization.py",
        "v2v": "scripts/run_v2v.py",
    }
    
    script = script_map.get(task)
    if not script:
        return False, f"⚠️  {task}: No runner script available"
    
    script_path = Path(__file__).parent.parent / script
    if not script_path.exists():
        return False, f"⚠️  {task}: Script not found: {script}"
    
    # Check if dataset exists
    dataset_path = Path(__file__).parent.parent / f"data/golden/{dataset}.yaml"
    if not dataset_path.exists():
        return False, f"⚠️  {task}/{dataset}: Dataset file not found"
    
    cmd = [
        sys.executable, str(script_path),
        "--model", model_id,
        "--dataset", dataset
    ]
    
    if dry_run:
        return True, f"[DRY-RUN] Would run: {' '.join(cmd)}"
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for model loading
        )
        
        if result.returncode == 0:
            return True, f"✅ {task}/{dataset}: PASSED"
        else:
            # Extract error message
            stderr = result.stderr[-500:] if result.stderr else ""
            stdout = result.stdout[-500:] if result.stdout else ""
            error_msg = stderr or stdout or "Unknown error"
            return False, f"❌ {task}/{dataset}: FAILED\n   {error_msg[:200]}"
            
    except subprocess.TimeoutExpired:
        return False, f"❌ {task}/{dataset}: TIMEOUT (10min)"
    except Exception as e:
        return False, f"❌ {task}/{dataset}: ERROR - {e}"


def regenerate_decisions() -> bool:
    """Regenerate arsenal and decisions."""
    try:
        decisions_script = Path(__file__).parent / "generate_decisions.py"
        subprocess.run([sys.executable, str(decisions_script)], check=True, capture_output=True)
        
        coverage_script = Path(__file__).parent / "coverage_report.py"
        if coverage_script.exists():
            subprocess.run([sys.executable, str(coverage_script)], check=True, capture_output=True)
        
        return True
    except Exception as e:
        print(f"Failed to regenerate decisions: {e}")


def init_claims(model_id: str, capabilities: List[str]) -> bool:
    """
    Initialize claims.yaml for a model based on its declared capabilities.
    
    Creates one required claim per capability with known test handlers.
    """
    claims_path = Path(f"models/{model_id}/claims.yaml")
    
    if claims_path.exists():
        print(f"  ✓ claims.yaml already exists: {claims_path}")
        return True
    
    # Claim templates per capability
    claim_templates = {
        "asr": {
            "id": f"{model_id}_asr_smoke_structural",
            "task": "asr",
            "type": "structural",
            "description": "ASR smoke run produces valid artifact with provenance",
            "enforcement": "required",
            "test_ref": "claims.asr.smoke_structural_v1",
            "thresholds": {
                "has_provenance": True,
                "has_run_context": True,
            }
        },
        "vad": {
            "id": f"{model_id}_vad_smoke_valid",
            "task": "vad",
            "type": "structural",
            "description": "VAD produces speech_ratio within sane bounds",
            "enforcement": "required",
            "test_ref": "claims.vad.smoke_ratio_v1",
            "thresholds": {
                "speech_ratio_min": 0.05,
                "speech_ratio_max": 0.95,
            }
        },
        "v2v": {
            "id": f"{model_id}_v2v_realtime",
            "task": "v2v",
            "type": "structural",
            "description": "V2V runs faster than real-time",
            "enforcement": "required",
            "test_ref": "claims.v2v.realtime_threshold_v1",
            "thresholds": {
                "rtf_like_max": 1.2,
            }
        },
        "diarization": {
            "id": f"{model_id}_diar_smoke_structural",
            "task": "diarization",
            "type": "structural",
            "description": "Diarization produces valid speaker segments",
            "enforcement": "required",
            "test_ref": "claims.diarization.smoke_structural_v1",
            "thresholds": {
                "has_speakers": True,
            }
        },
        "tts": {
            "id": f"{model_id}_tts_audio_output",
            "task": "tts",
            "type": "structural",
            "description": "TTS produces audio of non-trivial duration",
            "enforcement": "optional",  # Optional until MOS/quality gates
            "test_ref": "claims.tts.audio_output_v1",
            "thresholds": {
                "min_duration_s": 0.5,
            }
        },
        "chat": {
            "id": f"{model_id}_chat_responds",
            "task": "chat",
            "type": "structural",
            "description": "Chat returns non-empty response within budget",
            "enforcement": "optional",
            "test_ref": "claims.chat.responds_v1",
            "thresholds": {
                "response_non_empty": True,
                "max_latency_ms": 30000,
            }
        },
    }
    
    # Build claims list from capabilities
    claims = []
    for cap in capabilities:
        if cap in claim_templates:
            claims.append(claim_templates[cap])
    
    if not claims:
        print(f"  ⚠️ No claim templates for capabilities: {capabilities}")
        return False
    
    # Write claims.yaml
    claims_data = {
        "claims": claims
    }
    
    claims_path.parent.mkdir(parents=True, exist_ok=True)
    with open(claims_path, 'w') as f:
        yaml.dump(claims_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"  ✓ Created claims.yaml with {len(claims)} claims")
    return True


def get_eligible_use_cases(model_id: str) -> List[str]:
    """Determine which use cases this model is now eligible for."""
    import json
    
    decisions_path = Path(__file__).parent.parent / "docs" / "decisions.json"
    if not decisions_path.exists():
        return []
    
    try:
        with open(decisions_path) as f:
            decisions = json.load(f)
        
        eligible = []
        use_cases = decisions.get("use_cases", decisions.get("decisions", {}))
        for uc_id, uc_data in use_cases.items():
            pipeline = uc_data.get("pipeline", {})
            recommended = uc_data.get("recommended", [])
            if model_id in pipeline.values() or model_id in recommended:
                eligible.append(uc_id)
        
        return eligible
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(description="Onboard a new model to the arsenal")
    parser.add_argument("--model", required=True, help="Model ID to onboard")
    parser.add_argument("--tasks", help="Comma-separated tasks to run (default: all declared)")
    parser.add_argument("--golden", action="store_true", help="Also run golden_batch tests")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, don't run tests")
    parser.add_argument("--init-claims", action="store_true", help="Create claims.yaml skeleton")
    parser.add_argument("--no-claims", action="store_true", help="Skip claims verification")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"  Model Onboarding: {args.model}")
    print(f"{'='*60}\n")
    
    # Step 1: Validate config
    print("Step 1: Validating model config...")
    valid, errors, meta = validate_model_config(args.model)
    if not valid:
        print(f"❌ Config validation failed:")
        for e in errors:
            print(f"   - {e}")
        sys.exit(EXIT_CONFIG_INVALID)
    print("✅ Config loads correctly\n")
    
    # Step 2.1: Initialize claims if requested or always create skeleton
    capabilities = meta.get('capabilities', [])
    if args.init_claims or not Path(f"models/{args.model}/claims.yaml").exists():
        print("Step 2.1: Initializing claims skeleton...")
        init_claims(args.model, capabilities)
    
    # Step 2: Check environment requirements
    print("Step 2: Checking environment requirements...")
    env_ok, env_errors = check_env_requirements(args.model)
    if not env_ok:
        print(f"❌ Blocked by missing environment:")
        for e in env_errors:
            print(f"   - {e}")
        print(f"\n⚠️  No runs produced. Set required env vars and retry.")
        sys.exit(EXIT_MISSING_ENV)
    print("✅ Environment OK\n")
    
    # Step 3: Get declared capabilities
    capabilities = meta.get("capabilities", [])
    
    # Filter by --tasks if provided
    if args.tasks:
        requested = [t.strip() for t in args.tasks.split(",")]
        capabilities = [c for c in capabilities if c in requested]
        invalid_tasks = [t for t in requested if t not in meta.get("capabilities", [])]
        if invalid_tasks:
            print(f"⚠️  Tasks not declared by model: {invalid_tasks}")
    
    print(f"Step 3: Capabilities to test: {capabilities}\n")
    
    if not capabilities:
        print("❌ No capabilities to test")
        sys.exit(EXIT_CONFIG_INVALID)
    
    # Step 4: Run smoke tests for each capability
    print("Step 4: Running smoke tests for declared capabilities...")
    smoke_results = []
    
    for task in capabilities:
        datasets = get_smoke_datasets(task)
        if not datasets:
            print(f"   ⚠️  No smoke dataset for task: {task}")
            continue
            
        for dataset in datasets:
            passed, msg = run_smoke_test(args.model, task, dataset, args.dry_run)
            smoke_results.append((task, dataset, passed, msg))
            print(f"   {msg}")
    
    # Step 5: Optionally run golden tests
    golden_results = []
    if args.golden:
        print("\nStep 5: Running golden_batch tests...")
        for task in capabilities:
            datasets = get_golden_datasets(task)
            if not datasets:
                print(f"   ⚠️  No golden dataset for task: {task}")
                continue
                
            for dataset in datasets:
                passed, msg = run_smoke_test(args.model, task, dataset, args.dry_run)
                golden_results.append((task, dataset, passed, msg))
                print(f"   {msg}")
    
    # Step 6: Regenerate decisions  
    if not args.dry_run:
        print("\nStep 6: Regenerating decisions and coverage...")
        if regenerate_decisions():
            print("✅ Decisions and coverage regenerated\n")
        else:
            print("⚠️  Failed to regenerate decisions\n")
    
    # Step 7: Report eligibility
    print("Step 7: Checking use case eligibility...")
    if not args.dry_run:
        eligible = get_eligible_use_cases(args.model)
        if eligible:
            print(f"✅ Model is now eligible for: {', '.join(eligible)}")
        else:
            print("⚠️  Model not selected for any use case (may need golden evidence)")
    
    # Step 8: Verify claims
    claims_result = None
    if not args.no_claims and not args.dry_run:
        print("\nStep 8: Verifying claims...")
        claims_path = Path(f"models/{args.model}/claims.yaml")
        if not claims_path.exists():
            print(f"❌ No claims.yaml found. Use --init-claims or --no-claims")
            sys.exit(EXIT_CONFIG_INVALID)
        
        # Run claims via the executor
        try:
            from tests.claims.test_claims_execute import execute_all_claims, load_claims
            claims_data = load_claims(claims_path)
            claims_result = execute_all_claims(args.model, claims_data)
            
            # Display results
            for claim in claims_result['claims']:
                icon = "✅" if claim['status'] == 'pass' else "❌" if claim['status'] == 'fail' else "⏭️"
                enforcement = "[req]" if claim['enforcement'] == 'required' else "[opt]"
                print(f"  {icon} {enforcement} {claim['claim_id']}: {claim['reason']}")
            
            # Check for failures
            s = claims_result['summary']
            if s['enforced_fail'] > 0 or s['enforced_skip'] > 0:
                print(f"\n❌ Required claims failed: {s['enforced_fail']} fail, {s['enforced_skip']} skip")
                sys.exit(EXIT_SOME_FAILED)
            print(f"\n✅ Claims verified: {s['enforced_pass']} required pass")
        except Exception as e:
            print(f"⚠️ Claims verification error: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    
    smoke_passed = sum(1 for _, _, p, _ in smoke_results if p)
    smoke_total = len(smoke_results)
    print(f"Smoke tests: {smoke_passed}/{smoke_total} passed")
    
    if args.golden:
        golden_passed = sum(1 for _, _, p, _ in golden_results if p)
        golden_total = len(golden_results)
        print(f"Golden tests: {golden_passed}/{golden_total} passed")
    
    all_results = smoke_results + golden_results
    passed_count = sum(1 for _, _, p, _ in all_results if p)
    total_count = len(all_results)
    
    if total_count == 0:
        print("\n⚠️  No tests were run (no datasets available)")
        sys.exit(EXIT_CONFIG_INVALID)
    elif passed_count == 0:
        print("\n❌ All tests failed")
        sys.exit(EXIT_ALL_FAILED)
    elif passed_count < total_count:
        print(f"\n⚠️  Onboarding partial - {passed_count}/{total_count} tests passed")
        sys.exit(EXIT_SOME_FAILED)
    else:
        print(f"\n✅ Model {args.model} onboarded successfully!")
        
        # Machine-readable summary
        import json
        onboard_summary = {
            "model_id": args.model,
            "status": "SUCCESS",
            "tasks_attempted": [r[0] for r in all_results],
            "runs_written": passed_count,
            "quarantines": 0,
            "claims": claims_result['summary'] if claims_result else None
        }
        summary_path = Path(f"runs/{args.model}/onboard_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(onboard_summary, f, indent=2)
        print(f"\n📋 Summary: {summary_path}")
        
        sys.exit(EXIT_SUCCESS)


if __name__ == "__main__":
    main()
