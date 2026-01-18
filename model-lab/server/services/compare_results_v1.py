"""
Comparison Semantics Layer (v1)

Pure comparison of two ResultSummary objects.
See docs/compare_results_contract.md.
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

logger = logging.getLogger("server.compare")

def compute_comparison_v1(experiment_id: str, res_a: Optional[Dict], res_b: Optional[Dict]) -> Dict[str, Any]:
    """
    Computes ComparisonSummary between two results.
    """
    
    # Base Structure
    summary = {
        "schema_version": "v1",
        "experiment_id": experiment_id,
        "candidates": {
            "A": {"label": "Unknown", "run_id": None, "status": "UNKNOWN"},
            "B": {"label": "Unknown", "run_id": None, "status": "UNKNOWN"}
        },
        "readiness": {
            "comparable": False,
            "reason": "MISSING_RESULTS"
        },
        "metrics": {},
        "verdicts": {
            "overall": "PENDING",
            "reasons": []
        },
        "provenance": {
            "computed_at": datetime.now().isoformat(),
            "results_schema": "v1"
        }
    }

    # Populate Identity
    if res_a:
        summary["candidates"]["A"] = {
            "label": res_a.get("candidate_label", "Unknown"),
            "run_id": res_a.get("run_id"),
            "status": res_a.get("status")
        }
    if res_b:
        summary["candidates"]["B"] = {
            "label": res_b.get("candidate_label", "Unknown"),
            "run_id": res_b.get("run_id"),
            "status": res_b.get("status")
        }

    # Readiness Checks
    if not res_a or not res_b:
        summary["readiness"]["reason"] = "MISSING_RESULTS"
        return summary
    
    # Check Status (Must be TERMINAL)
    terms = ["COMPLETED", "FAILED"]
    status_a = res_a.get("status")
    status_b = res_b.get("status")
    
    if status_a not in terms or status_b not in terms:
        summary["readiness"]["reason"] = "NOT_TERMINAL"
        return summary

    # Comparable!
    summary["readiness"]["comparable"] = True
    summary["readiness"]["reason"] = None

    # Compute Metrics Deltas
    metrics = {}
    mets_a = res_a.get("metrics", {})
    mets_b = res_b.get("metrics", {})
    
    for key in ["word_count", "duration_s", "confidence_avg"]:
        val_a = mets_a.get(key)
        val_b = mets_b.get(key)
        
        if val_a is not None and val_b is not None:
             delta = val_a - val_b
             pct = 0.0
             if val_b != 0 and isinstance(val_b, (int, float)):
                 pct = (delta / val_b) * 100
             
             metrics[key] = {
                 "A": val_a,
                 "B": val_b,
                 "delta": delta,
                 "pct_change": pct
             }
    summary["metrics"] = metrics

    # Verdict Logic
    verdict = "INCONCLUSIVE"
    reasons = []

    # Case 1: Status mismatch
    if status_a == "COMPLETED" and status_b == "FAILED":
        verdict = "A_BETTER"
        reasons.append("A completed successfully while B failed.")
    elif status_b == "COMPLETED" and status_a == "FAILED":
        verdict = "B_BETTER"
        reasons.append("B completed successfully while A failed.")
    else:
        # Both COMPLETED or Both FAILED - Check Metrics
        
        # 1. Confidence (>5% significant)
        conf = metrics.get("confidence_avg")
        winner_declared = False
        
        if conf:
            delta_conf = conf["delta"] # A - B
            if delta_conf > 0.05:
                verdict = "A_BETTER"
                reasons.append(f"A has significantly higher confidence (+{conf['pct_change']:.1f}%).")
                winner_declared = True
            elif delta_conf < -0.05:
                verdict = "B_BETTER"
                reasons.append(f"B has significantly higher confidence (A is {conf['pct_change']:.1f}%).")
                winner_declared = True
        
        # 2. Word Count (>10% significant) - Only if no confidence winner
        if not winner_declared:
            wc = metrics.get("word_count")
            if wc:
                delta_wc_pct = wc["pct_change"]
                if delta_wc_pct > 10:
                    verdict = "A_BETTER"
                    reasons.append(f"A produced significantly more content (+{delta_wc_pct:.1f}%).")
                    winner_declared = True
                elif delta_wc_pct < -10:
                    verdict = "B_BETTER"
                    reasons.append(f"B produced significantly more content (A is {delta_wc_pct:.1f}%).")
                    winner_declared = True

        # 3. Duration (>10% significant) - Tie breaker
        if not winner_declared:
            dur = metrics.get("duration_s")
            if dur:
                delta_dur_pct = dur["pct_change"]
                if delta_dur_pct < -10: # A is faster (lower duration)
                     verdict = "A_BETTER"
                     reasons.append(f"A is significantly faster ({delta_dur_pct:.1f}%).")
                     winner_declared = True
                elif delta_dur_pct > 10: # B is faster
                     verdict = "B_BETTER"
                     reasons.append(f"B is significantly faster (A is +{delta_dur_pct:.1f}%).")
                     winner_declared = True
        
        if not winner_declared:
            verdict = "NEUTRAL"
            reasons.append("Difference is negligible across key metrics.")

    summary["verdicts"]["overall"] = verdict
    summary["verdicts"]["reasons"] = reasons
    
    return summary
