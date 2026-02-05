"""
Golden test set runner for regression testing.
Runs standardized tests against frozen datasets.
"""

import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
import logging

from .results import (
    RunResult, BatchResult, RunMetadata, PerformanceMetrics,
    ResultsManager, TaskType, ResultStatus, create_run_id
)
from .metrics_asr import ASRMetrics
from .metrics_entity import EntityMetrics
from .timers import PerformanceTimer
from .protocol import RunContract

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A single test case in the golden set."""
    id: str
    audio_path: Path
    ground_truth_path: Optional[Path] = None
    ground_truth_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def load_ground_truth(self) -> str:
        """Load ground truth text."""
        if self.ground_truth_text:
            return self.ground_truth_text
        if self.ground_truth_path and self.ground_truth_path.exists():
            return self.ground_truth_path.read_text().strip()
        return ""


@dataclass
class GoldenTestSet:
    """A frozen set of test cases for regression testing."""
    name: str
    version: str
    task_type: str
    test_cases: List[TestCase] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, filepath: Path) -> 'GoldenTestSet':
        """Load golden test set from YAML config."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        test_cases = []
        base_dir = filepath.parent
        
        for tc in data.get('test_cases', []):
            audio_path = base_dir / tc['audio_path']
            truth_path = base_dir / tc['ground_truth_path'] if tc.get('ground_truth_path') else None
            
            test_cases.append(TestCase(
                id=tc['id'],
                audio_path=audio_path,
                ground_truth_path=truth_path,
                ground_truth_text=tc.get('ground_truth_text'),
                metadata=tc.get('metadata', {}),
            ))
        
        return cls(
            name=data['name'],
            version=data['version'],
            task_type=data['task_type'],
            test_cases=test_cases,
            thresholds=data.get('thresholds', {}),
            metadata=data.get('metadata', {}),
        )
    
    def to_yaml(self, filepath: Path):
        """Save golden test set to YAML."""
        data = {
            'name': self.name,
            'version': self.version,
            'task_type': self.task_type,
            'thresholds': self.thresholds,
            'metadata': self.metadata,
            'test_cases': [
                {
                    'id': tc.id,
                    'audio_path': str(tc.audio_path.relative_to(filepath.parent)),
                    'ground_truth_path': str(tc.ground_truth_path.relative_to(filepath.parent)) if tc.ground_truth_path else None,
                    'ground_truth_text': tc.ground_truth_text,
                    'metadata': tc.metadata,
                }
                for tc in self.test_cases
            ],
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)


@dataclass
class RegressionResult:
    """Result of a regression test run."""
    passed: bool
    model_id: str
    golden_set: str
    timestamp: str
    batch_result: BatchResult
    threshold_checks: Dict[str, Dict[str, Any]]
    regressions: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'passed': self.passed,
            'model_id': self.model_id,
            'golden_set': self.golden_set,
            'timestamp': self.timestamp,
            'summary': self.batch_result.summary,
            'threshold_checks': self.threshold_checks,
            'regressions': self.regressions,
            'improvements': self.improvements,
        }


class GoldenTestRunner:
    """
    Runs golden test sets against models for regression testing.
    
    Usage:
        runner = GoldenTestRunner(runs_dir)
        result = runner.run(golden_set, model_id, transcribe_fn)
        if not result.passed:
            print("Regressions detected:", result.regressions)
    """
    
    def __init__(self, runs_dir: Path):
        self.runs_dir = Path(runs_dir)
        self.results_manager = ResultsManager(runs_dir)
        self.timer = PerformanceTimer()
    
    def run(self, 
            golden_set: GoldenTestSet,
            model_id: str,
            inference_fn: Callable,
            model_version: str = "latest",
            device: str = "cpu",
            compare_to_baseline: bool = True) -> RegressionResult:
        """
        Run golden test set against a model.
        
        Args:
            golden_set: The golden test set to run
            model_id: Model identifier
            inference_fn: Function(audio_path) -> (text, latency_s)
            model_version: Version string
            device: Device used for inference
            compare_to_baseline: Whether to compare against previous results
            
        Returns:
            RegressionResult with pass/fail status and details
        """
        batch_id = create_run_id()
        batch = BatchResult(
            batch_id=batch_id,
            model_id=model_id,
            task_type=golden_set.task_type,
        )
        
        logger.info(f"Running golden test set: {golden_set.name} v{golden_set.version}")
        logger.info(f"Model: {model_id}, {len(golden_set.test_cases)} test cases")
        
        for test_case in golden_set.test_cases:
            run_result = self._run_single(
                test_case=test_case,
                model_id=model_id,
                model_version=model_version,
                task_type=golden_set.task_type,
                inference_fn=inference_fn,
                device=device,
            )
            batch.add_result(run_result)
        
        # Check against thresholds
        threshold_checks = self._check_thresholds(batch, golden_set.thresholds)
        
        # Compare to baseline if requested
        regressions = []
        improvements = []
        if compare_to_baseline:
            baseline = self.results_manager.get_latest_batch(model_id, golden_set.task_type)
            if baseline:
                regressions, improvements = self._compare_to_baseline(batch, baseline)
        
        # Determine pass/fail
        passed = all(check['passed'] for check in threshold_checks.values())
        if regressions:
            passed = False
        
        # Save results
        self.results_manager.save_batch(batch)
        
        result = RegressionResult(
            passed=passed,
            model_id=model_id,
            golden_set=f"{golden_set.name}_v{golden_set.version}",
            timestamp=datetime.now().isoformat(),
            batch_result=batch,
            threshold_checks=threshold_checks,
            regressions=regressions,
            improvements=improvements,
        )
        
        self._log_result(result)
        return result
    
    def _run_single(self,
                    test_case: TestCase,
                    model_id: str,
                    model_version: str,
                    task_type: str,
                    inference_fn: Callable,
                    device: str) -> RunResult:
        """Run a single test case."""
        
        metadata = RunMetadata(
            run_id=f"{test_case.id}_{create_run_id()}",
            model_id=model_id,
            model_version=model_version,
            task_type=task_type,
            device=device,
        )
        
        ground_truth = test_case.load_ground_truth()
        errors = []
        
        try:
            # Run inference with timing
            memory_before = self.timer.get_memory_mb()
            
            import time
            start = time.perf_counter()
            output_text, extra_metrics = inference_fn(test_case.audio_path)
            latency_s = time.perf_counter() - start
            
            memory_after = self.timer.get_memory_mb()
            
            # Calculate metrics
            metrics = {}
            if task_type == TaskType.ASR.value and ground_truth:
                wer, s, d, i = ASRMetrics.calculate_wer(ground_truth, output_text)
                cer = ASRMetrics.calculate_cer(ground_truth, output_text)
                
                metrics = {
                    'wer': wer,
                    'cer': cer,
                    'substitutions': s,
                    'deletions': d,
                    'insertions': i,
                }
                
                # Entity error rate
                eer_result = EntityMetrics.calculate_eer(ground_truth, output_text)
                metrics['entity_error_rate'] = eer_result.entity_error_rate
            
            if extra_metrics:
                metrics.update(extra_metrics)
            
            # Get audio duration for RTF calculation
            import soundfile as sf
            info = sf.info(test_case.audio_path)
            audio_duration = info.duration
            
            performance = PerformanceMetrics(
                latency_ms=latency_s * 1000,
                rtf=latency_s / audio_duration if audio_duration > 0 else 0,
                memory_mb=memory_after - memory_before,
            )
            
            status = ResultStatus.SUCCESS.value
            
        except Exception as e:
            logger.error(f"Test case {test_case.id} failed: {e}")
            errors.append(str(e))
            output_text = None
            metrics = {}
            performance = PerformanceMetrics(latency_ms=0, rtf=0, memory_mb=0)
            status = ResultStatus.FAILED.value
        
        return RunResult(
            metadata=metadata,
            status=status,
            performance=performance,
            metrics=metrics,
            input_file=str(test_case.audio_path),
            output_text=output_text,
            ground_truth=ground_truth,
            errors=errors,
        )
    
    def _check_thresholds(self, batch: BatchResult, 
                          thresholds: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Check if batch results meet thresholds."""
        checks = {}
        summary = batch.summary
        
        for metric, threshold in thresholds.items():
            actual = summary.get(metric)
            if actual is None:
                checks[metric] = {
                    'passed': False,
                    'threshold': threshold,
                    'actual': None,
                    'reason': 'Metric not found in results',
                }
            else:
                # For error rates, lower is better
                if 'error' in metric or 'wer' in metric or 'cer' in metric:
                    passed = actual <= threshold
                else:
                    passed = actual >= threshold
                
                checks[metric] = {
                    'passed': passed,
                    'threshold': threshold,
                    'actual': actual,
                    'delta': actual - threshold,
                }
        
        return checks
    
    def _compare_to_baseline(self, current: BatchResult, 
                             baseline: BatchResult) -> tuple:
        """Compare current results to baseline."""
        regressions = []
        improvements = []
        
        current_summary = current.summary
        baseline_summary = baseline.summary
        
        # Compare WER
        if 'wer_mean' in current_summary and 'wer_mean' in baseline_summary:
            delta = current_summary['wer_mean'] - baseline_summary['wer_mean']
            if delta > 0.01:  # 1% regression threshold
                regressions.append(
                    f"WER regression: {baseline_summary['wer_mean']:.3f} -> {current_summary['wer_mean']:.3f} (+{delta:.3f})"
                )
            elif delta < -0.01:
                improvements.append(
                    f"WER improvement: {baseline_summary['wer_mean']:.3f} -> {current_summary['wer_mean']:.3f} ({delta:.3f})"
                )
        
        # Compare latency
        if 'latency_mean_ms' in current_summary and 'latency_mean_ms' in baseline_summary:
            delta_pct = (current_summary['latency_mean_ms'] - baseline_summary['latency_mean_ms']) / baseline_summary['latency_mean_ms']
            if delta_pct > 0.2:  # 20% regression threshold
                regressions.append(
                    f"Latency regression: {baseline_summary['latency_mean_ms']:.0f}ms -> {current_summary['latency_mean_ms']:.0f}ms (+{delta_pct*100:.0f}%)"
                )
            elif delta_pct < -0.2:
                improvements.append(
                    f"Latency improvement: {baseline_summary['latency_mean_ms']:.0f}ms -> {current_summary['latency_mean_ms']:.0f}ms ({delta_pct*100:.0f}%)"
                )
        
        return regressions, improvements
    
    def _log_result(self, result: RegressionResult):
        """Log regression test result."""
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        logger.info(f"Regression test {status}: {result.golden_set}")
        
        for metric, check in result.threshold_checks.items():
            if check['passed']:
                logger.info(f"  ✓ {metric}: {check['actual']:.3f} (threshold: {check['threshold']:.3f})")
            else:
                logger.warning(f"  ✗ {metric}: {check['actual']} (threshold: {check['threshold']})")
        
        for regression in result.regressions:
            logger.warning(f"  ⚠️ {regression}")
        
        for improvement in result.improvements:
            logger.info(f"  📈 {improvement}")
