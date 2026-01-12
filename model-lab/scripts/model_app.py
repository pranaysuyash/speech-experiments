#!/usr/bin/env python3
"""
Model App CLI - Canonical entry point for the Model Lab.

Exit Codes:
    0 - Success
    1 - General error
    2 - Missing evidence
    3 - Operability failed
    4 - Runner failed
    5 - Invalid artifact (provenance/run_context missing)

Usage:
    python scripts/model_app.py status [--json]
    python scripts/model_app.py recommend [--use-case ID] [--json]
    python scripts/model_app.py coverage [--json]
    python scripts/model_app.py decisions [--json]
    python scripts/model_app.py onboard --model <id> [--tasks asr,vad]
    python scripts/model_app.py run --task asr --model faster_whisper --dataset primary
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Exit codes
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_MISSING_EVIDENCE = 2
EXIT_OPERABILITY_FAILED = 3
EXIT_RUNNER_FAILED = 4
EXIT_INVALID_ARTIFACT = 5


def cmd_recommend(args):
    """Recommend best model for a task or use case."""
    # Load decisions
    decisions_path = Path(__file__).parent.parent / "docs" / "decisions.json"
    if not decisions_path.exists():
        if args.json:
            print(json.dumps({"error": "decisions.json not found"}))
        else:
            print("❌ decisions.json not found. Run: model_app.py decisions")
        sys.exit(EXIT_MISSING_EVIDENCE)
    
    with open(decisions_path) as f:
        decisions = json.load(f)
    
    use_cases = decisions.get("use_cases", {})
    tasks = decisions.get("tasks", {})
    
    # Task-based recommendation (contract test path)
    if getattr(args, 'task', None):
        task_data = tasks.get(args.task)
        if not task_data:
            if args.json:
                print(json.dumps({"error": f"Unknown task: {args.task}", "available": list(tasks.keys())}))
            else:
                print(f"❌ Unknown task: {args.task}")
                print(f"Available tasks: {list(tasks.keys())}")
            sys.exit(EXIT_ERROR)
        
        # Get best model for this task
        best = task_data.get("best_by_outcome") or task_data.get("best")
        outcome = "RECOMMENDED" if best else "unknown"
        models = task_data.get("models", [])
        
        # Filter by device if specified
        device_filter = getattr(args, 'device', None)
        if device_filter and models:
            # Filter to only models with matching device evidence
            device_models = [m for m in models if m.get('device') == device_filter]
            if not device_models:
                if args.json:
                    print(json.dumps({
                        "error": f"No evidence for device: {device_filter}",
                        "available_devices": list(set(m.get('device') for m in models if m.get('device')))
                    }))
                else:
                    print(f"\n⚠️ No evidence for device: {device_filter}")
                    available = list(set(m.get('device') for m in models if m.get('device')))
                    print(f"Available devices: {available}")
                sys.exit(EXIT_MISSING_EVIDENCE)
            models = device_models
            best = models[0].get('model_id') if models else None
            outcome = "RECOMMENDED" if best else "unknown"
        
        if args.json:
            result = {
                "task": args.task,
                "best_model": best,
                "outcome": outcome,
                "candidates": models[:3],  # Top 3
                "device_filter": device_filter,
            }
            # If audio provided, indicate it would run
            if getattr(args, 'audio', None):
                result["will_run"] = True
                result["audio"] = str(args.audio)
            print(json.dumps(result, indent=2))
        else:
            icon = "✅" if outcome.upper() == "RECOMMENDED" else "⚠️" if outcome.upper() == "ACCEPTABLE" else "❌"
            print(f"\nTask: {args.task}")
            print(f"Best Model: {best or 'None'}")
            print(f"Outcome: {icon} {outcome}")
            if device_filter:
                print(f"Device: {device_filter}")
            if models:
                print("\nTop candidates:")
                for i, m in enumerate(models[:3], 1):
                    device = m.get('device', 'unknown')
                    print(f"  {i}. {m.get('model_id')} (grade={m.get('evidence_grade')}, device={device})")
        
        # If audio file provided, run the model
        if getattr(args, 'audio', None) and best:
            audio_path = Path(args.audio)
            if not audio_path.exists():
                print(f"\n❌ Audio file not found: {audio_path}")
                sys.exit(EXIT_ERROR)
            
            # Validate --prompt usage (v2v only)
            if getattr(args, 'prompt', None) and args.task != "v2v":
                print(f"\n❌ --prompt is only valid for 'v2v' task, not '{args.task}'")
                sys.exit(EXIT_ERROR)
            
            print(f"\n🎵 Running {best} on {audio_path.name}...")
            script_map = {
                "asr": "scripts/run_asr.py",
                "vad": "scripts/run_vad.py",
                "diarization": "scripts/run_diarization.py",
                "v2v": "scripts/run_v2v.py",
            }
            script = script_map.get(args.task)
            if script:
                # Run with adhoc dataset (user-provided audio)
                cmd = [sys.executable, str(Path(__file__).parent.parent / script),
                       "--model", best, "--audio", str(audio_path.resolve())]
                
                # Add --prompt for v2v only
                if args.task == "v2v" and getattr(args, 'prompt', None):
                    cmd.extend(["--prompt", args.prompt])
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Parse ARTIFACT_PATH from runner output
                artifact_path = None
                for line in (result.stdout or '').split('\n'):
                    if line.startswith("ARTIFACT_PATH:"):
                        artifact_path = line.split(":", 1)[1].strip()
                        break
                
                if result.returncode == 0:
                    print(f"✅ Run completed")
                    if artifact_path:
                        print(f"📄 Artifact: {artifact_path}")
                else:
                    print(f"❌ Run failed")
                    if result.stderr:
                        print(f"   Error: {result.stderr[-200:]}")
            else:
                print(f"❌ Cannot run task '{args.task}' with audio directly.")
        
        return # Exit after handling task-based recommendation
    
    # Original use-case based recommendation
    if not decisions_path.exists():
        if args.json:
            print(json.dumps({"error": "decisions.json not found"}))
        else:
            print("❌ decisions.json not found. Run: model_app.py decisions")
        sys.exit(EXIT_MISSING_EVIDENCE)
    
    # decisions already loaded above
    # use_cases = decisions.get("use_cases", decisions.get("decisions", {})) # This line is now redundant
    
    if args.json:
        if args.use_case:
            uc_data = use_cases.get(args.use_case)
            if uc_data:
                print(json.dumps({"use_case": args.use_case, "data": uc_data}, indent=2))
            else:
                print(json.dumps({"error": f"Unknown use case: {args.use_case}"}))
                sys.exit(EXIT_ERROR)
        else:
            print(json.dumps({"use_cases": use_cases}, indent=2))
        return
    
    if args.use_case:
        uc_data = use_cases.get(args.use_case)
        if not uc_data:
            print(f"❌ Unknown use case: {args.use_case}")
            print(f"Available: {list(use_cases.keys())}")
            sys.exit(EXIT_ERROR)
        
        outcome = uc_data.get("outcome", "RECOMMENDED" if uc_data.get("recommended") else "unknown")
        pipeline = uc_data.get("pipeline", {})
        recommended = uc_data.get("recommended", [])
        
        print(f"\nUse Case: {args.use_case}")
        print(f"Outcome:  {outcome}")
        
        if pipeline:
            print("\nRecommended Pipeline:")
            for task, model in pipeline.items():
                print(f"  {task}: {model}")
        elif recommended:
            print(f"\nRecommended Models: {', '.join(recommended)}")
        else:
            print("\nNo pipeline available")
            if uc_data.get("fatal_reasons"):
                print("\nBlockers:")
                for reason in uc_data["fatal_reasons"]:
                    print(f"  - {reason}")
    else:
        print("\nAvailable Use Cases:")
        print("-" * 50)
        for uc_id, uc_data in use_cases.items():
            outcome = uc_data.get("outcome")
            if not outcome:
                outcome = "RECOMMENDED" if uc_data.get("recommended") else "REJECTED"
            icon = "✅" if outcome.upper() == "RECOMMENDED" else "⚠️" if outcome.upper() == "ACCEPTABLE" else "❌"
            print(f"{icon} {uc_id}: {outcome}")


def cmd_run(args):
    """Run a specific model on a task/dataset."""
    script_map = {
        "asr": "scripts/run_asr.py",
        "vad": "scripts/run_vad.py",
        "diarization": "scripts/run_diarization.py",
        "v2v": "scripts/run_v2v.py",
    }
    
    if args.task not in script_map:
        print(f"❌ Unknown task: {args.task}")
        print(f"Available: {list(script_map.keys())}")
        sys.exit(EXIT_ERROR)
    
    script = Path(__file__).parent.parent / script_map[args.task]
    
    cmd = [sys.executable, str(script), "--model", args.model]
    
    if args.dataset:
        cmd.extend(["--dataset", args.dataset])
    if args.device:
        cmd.extend(["--device", args.device])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        sys.exit(EXIT_RUNNER_FAILED)
    sys.exit(EXIT_SUCCESS)


def cmd_test(args):
    """Run smoke tests for a model's declared capabilities."""
    onboard_script = Path(__file__).parent / "onboard_model.py"
    
    cmd = [sys.executable, str(onboard_script), "--model", args.model]
    
    if args.golden:
        cmd.append("--golden")
    if args.dry_run:
        cmd.append("--dry-run")
    
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def cmd_decisions(args):
    """Regenerate decisions and print summary."""
    decisions_script = Path(__file__).parent / "generate_decisions.py"
    
    if not args.json:
        print("Regenerating decisions...")
    
    result = subprocess.run([sys.executable, str(decisions_script)], capture_output=True, text=True)
    
    if result.returncode != 0:
        if args.json:
            print(json.dumps({"error": result.stderr}))
        else:
            print(f"❌ Failed: {result.stderr}")
        sys.exit(EXIT_RUNNER_FAILED)
    
    decisions_path = Path(__file__).parent.parent / "docs" / "decisions.json"
    with open(decisions_path) as f:
        decisions = json.load(f)
    
    if args.json:
        print(json.dumps(decisions, indent=2))
        return
    
    use_cases = decisions.get("use_cases", decisions.get("decisions", {}))
    print("\nDecisions Summary:")
    print("-" * 50)
    for uc_id, uc_data in use_cases.items():
        outcome = uc_data.get("outcome")
        if not outcome:
            outcome = "RECOMMENDED" if uc_data.get("recommended") else "REJECTED"
        icon = "✅" if outcome == "RECOMMENDED" else "⚠️" if outcome == "ACCEPTABLE" else "❌"
        print(f"{icon} {uc_id}: {outcome}")


def cmd_coverage(args):
    """Generate and display coverage report."""
    coverage_script = Path(__file__).parent / "coverage_report.py"
    
    result = subprocess.run([sys.executable, str(coverage_script)], capture_output=True, text=True)
    
    if result.returncode != 0:
        if args.json:
            print(json.dumps({"error": result.stderr}))
        else:
            print(f"❌ Failed: {result.stderr}")
        sys.exit(EXIT_RUNNER_FAILED)
    
    if args.json:
        coverage_path = Path(__file__).parent.parent / "docs" / "coverage.json"
        if coverage_path.exists():
            with open(coverage_path) as f:
                print(f.read())
        return
    
    # Print human output
    print(result.stdout)
    print("=" * 60)
    print("See full report: docs/COVERAGE.md")


def cmd_onboard(args):
    """Onboard a new model."""
    onboard_script = Path(__file__).parent / "onboard_model.py"
    
    cmd = [sys.executable, str(onboard_script), "--model", args.model]
    
    if args.golden:
        cmd.append("--golden")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.tasks:
        cmd.extend(["--tasks", args.tasks])
    
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def cmd_status(args):
    """Show quick status: models, evidence, decisions."""
    from harness.registry import ModelRegistry
    
    registry = ModelRegistry()
    models = registry.list_models()
    
    # Collect status data
    status_data = {
        "models": {},
        "evidence": {},
        "decisions": {}
    }
    
    for m in models:
        meta = registry.get_model_metadata(m)
        status_data["models"][m] = {
            "status": meta.get("status", "unknown"),
            "capabilities": meta.get("capabilities", [])
        }
    
    runs_dir = Path(__file__).parent.parent / "runs"
    quarantine_dir = Path(__file__).parent.parent / "runs_quarantine"
    
    status_data["evidence"]["valid_runs"] = len(list(runs_dir.glob("**/*.json"))) if runs_dir.exists() else 0
    status_data["evidence"]["quarantined"] = len(list(quarantine_dir.glob("**/*.json"))) if quarantine_dir.exists() else 0
    
    decisions_path = Path(__file__).parent.parent / "docs" / "decisions.json"
    if decisions_path.exists():
        with open(decisions_path) as f:
            decisions = json.load(f)
        use_cases = decisions.get("use_cases", decisions.get("decisions", {}))
        for uc_id, uc_data in use_cases.items():
            outcome = uc_data.get("outcome")
            if not outcome:
                outcome = "RECOMMENDED" if uc_data.get("recommended") else "REJECTED"
            status_data["decisions"][uc_id] = outcome
    
    if args.json:
        print(json.dumps(status_data, indent=2))
        return
    
    print("\n📊 Model Lab Status")
    print("=" * 50)
    
    print(f"\nRegistered Models: {len(models)}")
    for m, data in status_data["models"].items():
        caps = ", ".join(data["capabilities"]) if data["capabilities"] else "none"
        print(f"  • {m} ({data['status']}): {caps}")
    
    print(f"\nEvidence:")
    print(f"  • Valid runs: {status_data['evidence']['valid_runs']}")
    print(f"  • Quarantined: {status_data['evidence']['quarantined']}")
    
    if status_data["decisions"]:
        print(f"\nDecisions:")
        for uc_id, outcome in status_data["decisions"].items():
            icon = "✅" if outcome == "RECOMMENDED" else "⚠️" if outcome == "ACCEPTABLE" else "❌"
            print(f"  {icon} {uc_id}: {outcome}")
    else:
        print("\n⚠️ No decisions generated yet")
    
    print()


def cmd_claims(args):
    """Verify claims for a model."""
    import yaml
    from tests.claims.test_claims_execute import execute_all_claims, load_claims
    
    models_dir = Path(__file__).parent.parent / "models"
    
    if args.model:
        # Single model
        claims_path = models_dir / args.model / "claims.yaml"
        if not claims_path.exists():
            if args.json:
                print(json.dumps({"error": f"No claims.yaml for {args.model}"}))
            else:
                print(f"❌ No claims.yaml found for: {args.model}")
            sys.exit(EXIT_MISSING_EVIDENCE)
        
        data = load_claims(claims_path)
        results = execute_all_claims(args.model, data)
        
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"\n📋 Claims for {args.model}")
            print("=" * 50)
            
            for claim in results['claims']:
                icon = "✅" if claim['status'] == 'pass' else "❌" if claim['status'] == 'fail' else "⏭️"
                enforcement = "[req]" if claim['enforcement'] == 'required' else "[opt]"
                print(f"  {icon} {enforcement} {claim['claim_id']}: {claim['reason']}")
            
            s = results['summary']
            print(f"\nSummary:")
            print(f"  Enforced: {s['enforced_pass']} pass, {s['enforced_fail']} fail, {s['enforced_skip']} skip")
            print(f"  Optional: {s['optional_pass']} pass, {s['optional_fail']} fail, {s['optional_skip']} skip")
        
        # Exit code based on enforced failures
        if results['summary']['enforced_fail'] > 0:
            sys.exit(EXIT_OPERABILITY_FAILED)
        sys.exit(EXIT_SUCCESS)
    else:
        # List all models with claims
        all_claims = list(models_dir.glob("*/claims.yaml"))
        
        if args.json:
            all_results = {}
            for claims_path in all_claims:
                model_id = claims_path.parent.name
                data = load_claims(claims_path)
                all_results[model_id] = execute_all_claims(model_id, data)
            print(json.dumps(all_results, indent=2))
        else:
            print("\n📋 Models with claims:")
            for claims_path in all_claims:
                print(f"  • {claims_path.parent.name}")
            print(f"\nUse: model_app.py claims --model <id> to verify")


def cmd_sweep(args):
    """Run tests for all models missing evidence at target grade."""
    import os
    from harness.registry import ModelRegistry
    
    registry = ModelRegistry()
    models = registry.list_models()
    
    # Load coverage to find gaps
    coverage_path = Path(__file__).parent.parent / "docs" / "coverage.json"
    if not coverage_path.exists():
        # Generate coverage first
        subprocess.run([sys.executable, str(Path(__file__).parent / "coverage_report.py")], 
                       capture_output=True)
    
    with open(coverage_path) as f:
        coverage = json.load(f)
    
    results = {
        "success": [],
        "failed": [],
        "blocked": [],
        "skipped": []
    }
    
    # ENV requirements
    env_requirements = {
        "pyannote_diarization": ["HF_TOKEN"],
    }
    
    # Smoke dataset mapping
    smoke_datasets = {
        "asr": "asr_smoke_v1",
        "vad": "vad_smoke_v1", 
        "diarization": "diar_smoke_v1",
        "v2v": "v2v_smoke_v1",
    }
    
    # Golden dataset mapping  
    golden_datasets = {
        "asr": "primary",
    }
    
    target_grade = args.grade or "smoke"
    datasets = golden_datasets if target_grade == "golden_batch" else smoke_datasets
    
    print(f"\n🔄 Sweep: filling evidence gaps at grade={target_grade}")
    print("=" * 60)
    
    for model_id in models:
        meta = registry.get_model_metadata(model_id)
        capabilities = meta.get("capabilities", [])
        
        # Filter by task if specified (unless --all-tasks)
        if args.task and not getattr(args, 'all_tasks', False):
            capabilities = [c for c in capabilities if c == args.task]
        
        if not capabilities:
            continue
            
        # Check env requirements
        required_env = env_requirements.get(model_id, [])
        missing_env = [v for v in required_env if not os.environ.get(v)]
        if missing_env:
            results["blocked"].append({
                "model": model_id,
                "reason": f"Missing env: {', '.join(missing_env)}"
            })
            print(f"  ⛔ {model_id}: blocked (missing {', '.join(missing_env)})")
            continue
        
        # Check each declared capability
        model_cov = coverage.get("models", {}).get(model_id, {})
        tasks_cov = model_cov.get("tasks", {})
        
        for task in capabilities:
            task_cov = tasks_cov.get(task, {})
            
            # Check if evidence exists at target grade
            has_evidence = task_cov.get("has_evidence", False)
            best_grade = task_cov.get("best_grade", "")
            
            grade_rank = {"golden_batch": 3, "smoke": 2, "adhoc": 1}
            target_rank = grade_rank.get(target_grade, 2)
            current_rank = grade_rank.get(best_grade, 0)
            
            if current_rank >= target_rank:
                results["skipped"].append({"model": model_id, "task": task, "reason": f"already has {best_grade}"})
                continue
            
            # Get dataset
            dataset = datasets.get(task)
            if not dataset:
                results["skipped"].append({"model": model_id, "task": task, "reason": "no dataset"})
                continue
            
            # Run the test
            print(f"  → {model_id}/{task} ({dataset})...", end=" ", flush=True)
            
            script_map = {
                "asr": "scripts/run_asr.py",
                "vad": "scripts/run_vad.py",
                "diarization": "scripts/run_diarization.py",
                "v2v": "scripts/run_v2v.py",
            }
            
            script = script_map.get(task)
            if not script:
                print("no runner")
                results["skipped"].append({"model": model_id, "task": task, "reason": "no runner"})
                continue
            
            script_path = Path(__file__).parent.parent / script
            cmd = [sys.executable, str(script_path), "--model", model_id, "--dataset", dataset]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    print("✅")
                    results["success"].append({"model": model_id, "task": task})
                else:
                    print("❌")
                    results["failed"].append({"model": model_id, "task": task, "error": result.stderr[-200:]})
            except subprocess.TimeoutExpired:
                print("⏱️ timeout")
                results["failed"].append({"model": model_id, "task": task, "error": "timeout"})
            except Exception as e:
                print(f"❌ {e}")
                results["failed"].append({"model": model_id, "task": task, "error": str(e)})
    
    # Regenerate coverage and decisions
    print("\n📊 Regenerating coverage and decisions...")
    subprocess.run([sys.executable, str(Path(__file__).parent / "coverage_report.py")], capture_output=True)
    subprocess.run([sys.executable, str(Path(__file__).parent / "generate_decisions.py")], capture_output=True)
    
    # Summary
    print(f"\n{'='*60}")
    print("  Sweep Summary")
    print(f"{'='*60}")
    print(f"✅ Success: {len(results['success'])}")
    print(f"❌ Failed: {len(results['failed'])}")
    print(f"⛔ Blocked: {len(results['blocked'])}")
    print(f"⏭️  Skipped: {len(results['skipped'])}")
    
    if args.json:
        print("\n" + json.dumps(results, indent=2))
    
    # Write sweep_report.json
    report_path = Path(__file__).parent.parent / "docs" / "sweep_report.json"
    report = {
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "grade": target_grade,
        **results,
        "summary": {
            "success": len(results['success']),
            "failed": len(results['failed']),
            "blocked": len(results['blocked']),
            "skipped": len(results['skipped']),
        }
    }
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n📋 Report: {report_path}")
    
    # Exit code: 0 even if some blocked
    if results["failed"]:
        sys.exit(EXIT_SOME_FAILED)
    sys.exit(EXIT_SUCCESS)


# Add missing constant
EXIT_SOME_FAILED = 3


def main():
    parser = argparse.ArgumentParser(
        description="Model Lab CLI - One door for all model operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
  0 - Success
  2 - Missing evidence
  3 - Operability failed
  4 - Runner failed
  5 - Invalid artifact

Examples:
  model_app.py status --json              # JSON output
  model_app.py recommend                  # List all use cases
  model_app.py coverage --json            # Machine-readable coverage
  model_app.py onboard --model whisper    # Onboard new model
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # recommend
    p_recommend = subparsers.add_parser("recommend", help="Get recommendation for use case or task")
    p_recommend.add_argument("--use-case", "-u", help="Use case ID")
    p_recommend.add_argument("--task", "-t", help="Task: asr, vad, diarization, v2v")
    p_recommend.add_argument("--audio", "-a", type=Path, help="Audio file to run on")
    p_recommend.add_argument("--prompt", "-p", help="Text prompt for V2V (only valid with --task v2v)")
    p_recommend.add_argument("--device", help="Filter by device: cpu, mps, cuda")
    p_recommend.add_argument("--json", action="store_true", help="Output JSON")
    p_recommend.set_defaults(func=cmd_recommend)
    
    # run
    p_run = subparsers.add_parser("run", help="Run model on task/dataset")
    p_run.add_argument("--task", "-t", required=True, help="Task: asr, vad, diarization, v2v")
    p_run.add_argument("--model", "-m", required=True, help="Model ID")
    p_run.add_argument("--dataset", "-d", help="Dataset ID")
    p_run.add_argument("--device", help="Device: cpu, mps, cuda")
    p_run.set_defaults(func=cmd_run)
    
    # test
    p_test = subparsers.add_parser("test", help="Run smoke tests for model")
    p_test.add_argument("--model", "-m", required=True, help="Model ID")
    p_test.add_argument("--golden", action="store_true", help="Also run golden tests")
    p_test.add_argument("--dry-run", action="store_true", help="Validate only")
    p_test.set_defaults(func=cmd_test)
    
    # decisions
    p_decisions = subparsers.add_parser("decisions", help="Regenerate decisions")
    p_decisions.add_argument("--json", action="store_true", help="Output JSON")
    p_decisions.set_defaults(func=cmd_decisions)
    
    # coverage
    p_coverage = subparsers.add_parser("coverage", help="Generate coverage report")
    p_coverage.add_argument("--json", action="store_true", help="Output JSON")
    p_coverage.set_defaults(func=cmd_coverage)
    
    # onboard
    p_onboard = subparsers.add_parser("onboard", help="Onboard new model")
    p_onboard.add_argument("--model", "-m", required=True, help="Model ID")
    p_onboard.add_argument("--tasks", help="Comma-separated tasks to run (default: all declared)")
    p_onboard.add_argument("--golden", action="store_true", help="Also run golden tests")
    p_onboard.add_argument("--dry-run", action="store_true", help="Validate only")
    p_onboard.set_defaults(func=cmd_onboard)
    
    # status
    p_status = subparsers.add_parser("status", help="Quick status overview")
    p_status.add_argument("--json", action="store_true", help="Output JSON")
    p_status.set_defaults(func=cmd_status)
    
    # sweep
    p_sweep = subparsers.add_parser("sweep", help="Run tests for all models missing evidence")
    p_sweep.add_argument("--grade", choices=["smoke", "golden_batch"], default="smoke",
                         help="Target evidence grade (default: smoke)")
    p_sweep.add_argument("--task", "-t", help="Filter to specific task (asr, vad, etc.)")
    p_sweep.add_argument("--all-tasks", action="store_true", dest="all_tasks",
                         help="Run all declared tasks that have runners")
    p_sweep.add_argument("--max-models", type=int, default=100, help="Max models to process (default: 100)")
    p_sweep.add_argument("--max-runs", type=int, default=50, help="Max runs to create (default: 50)")
    p_sweep.add_argument("--json", action="store_true", help="Output JSON results")
    p_sweep.set_defaults(func=cmd_sweep)
    
    # claims
    p_claims = subparsers.add_parser("claims", help="Verify model claims")
    p_claims.add_argument("--model", "-m", help="Model ID (omit to list all)")
    p_claims.add_argument("--json", action="store_true", help="Output JSON")
    p_claims.set_defaults(func=cmd_claims)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(EXIT_SUCCESS)
    
    args.func(args)


if __name__ == "__main__":
    main()
