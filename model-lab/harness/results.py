"""
Standardized result schema for model evaluation runs.
Ensures consistent JSON output for comparison across models.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum


class TaskType(str, Enum):
    ASR = "asr"
    TTS = "tts"
    CHAT = "chat"
    TRANSLATION = "translation"


class ResultStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class RunMetadata:
    """Metadata for a single evaluation run."""
    run_id: str
    model_id: str
    model_version: str
    task_type: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    device: str = "cpu"
    git_hash: Optional[str] = None
    config_hash: Optional[str] = None
    dataset_hash: Optional[str] = None
    notes: str = ""


@dataclass
class PerformanceMetrics:
    """Performance metrics for a run."""
    latency_ms: float
    rtf: float  # Real-time factor
    memory_mb: float
    first_token_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None


@dataclass
class ASRMetricsResult:
    """ASR-specific metrics."""
    wer: float
    cer: float
    entity_error_rate: Optional[float] = None
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    word_count_ref: int = 0
    word_count_hyp: int = 0


@dataclass
class TTSMetricsResult:
    """TTS-specific metrics."""
    similarity_score: float
    duration_ratio: float
    spectral_centroid: Optional[float] = None
    rms_energy: Optional[float] = None


@dataclass
class RunResult:
    """Complete result for a single evaluation run."""
    metadata: RunMetadata
    status: str
    performance: PerformanceMetrics
    metrics: Dict[str, Any]
    input_file: str
    output_text: Optional[str] = None
    output_audio_path: Optional[str] = None
    ground_truth: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'metadata': asdict(self.metadata),
            'status': self.status,
            'performance': asdict(self.performance),
            'metrics': self.metrics,
            'input_file': self.input_file,
            'output_text': self.output_text,
            'output_audio_path': self.output_audio_path,
            'ground_truth': self.ground_truth,
            'errors': self.errors,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunResult':
        """Create from dictionary."""
        return cls(
            metadata=RunMetadata(**data['metadata']),
            status=data['status'],
            performance=PerformanceMetrics(**data['performance']),
            metrics=data['metrics'],
            input_file=data['input_file'],
            output_text=data.get('output_text'),
            output_audio_path=data.get('output_audio_path'),
            ground_truth=data.get('ground_truth'),
            errors=data.get('errors', []),
        )


@dataclass
class BatchResult:
    """Results for a batch of evaluations."""
    batch_id: str
    model_id: str
    task_type: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    results: List[RunResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, result: RunResult):
        """Add a single result to the batch."""
        self.results.append(result)
        self._update_summary()
    
    def _update_summary(self):
        """Update summary statistics."""
        if not self.results:
            return
        
        successful = [r for r in self.results if r.status == ResultStatus.SUCCESS.value]
        
        self.summary = {
            'total_runs': len(self.results),
            'successful': len(successful),
            'failed': len(self.results) - len(successful),
            'success_rate': len(successful) / len(self.results) if self.results else 0,
        }
        
        if successful:
            latencies = [r.performance.latency_ms for r in successful]
            self.summary['latency_mean_ms'] = sum(latencies) / len(latencies)
            self.summary['latency_min_ms'] = min(latencies)
            self.summary['latency_max_ms'] = max(latencies)
            
            # Task-specific aggregation
            if self.task_type == TaskType.ASR.value:
                wers = [r.metrics.get('wer', 0) for r in successful if 'wer' in r.metrics]
                if wers:
                    self.summary['wer_mean'] = sum(wers) / len(wers)
                    self.summary['wer_min'] = min(wers)
                    self.summary['wer_max'] = max(wers)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'batch_id': self.batch_id,
            'model_id': self.model_id,
            'task_type': self.task_type,
            'timestamp': self.timestamp,
            'results': [r.to_dict() for r in self.results],
            'summary': self.summary,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchResult':
        """Create from dictionary."""
        batch = cls(
            batch_id=data['batch_id'],
            model_id=data['model_id'],
            task_type=data['task_type'],
            timestamp=data['timestamp'],
            summary=data.get('summary', {}),
        )
        batch.results = [RunResult.from_dict(r) for r in data.get('results', [])]
        return batch


class ResultsManager:
    """Manage saving and loading of evaluation results."""
    
    def __init__(self, runs_dir: Path):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
    
    def save_result(self, result: RunResult) -> Path:
        """Save a single run result."""
        model_dir = self.runs_dir / result.metadata.model_id / result.metadata.task_type
        model_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{result.metadata.run_id}.json"
        filepath = model_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        return filepath
    
    def save_batch(self, batch: BatchResult) -> Path:
        """Save a batch of results."""
        model_dir = self.runs_dir / batch.model_id / batch.task_type
        model_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"batch_{batch.batch_id}.json"
        filepath = model_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(batch.to_dict(), f, indent=2)
        
        return filepath
    
    def load_result(self, filepath: Path) -> RunResult:
        """Load a single run result."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return RunResult.from_dict(data)
    
    def load_batch(self, filepath: Path) -> BatchResult:
        """Load a batch of results."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return BatchResult.from_dict(data)
    
    def list_results(self, model_id: Optional[str] = None, 
                     task_type: Optional[str] = None) -> List[Path]:
        """List all result files matching criteria."""
        if model_id and task_type:
            search_dir = self.runs_dir / model_id / task_type
        elif model_id:
            search_dir = self.runs_dir / model_id
        else:
            search_dir = self.runs_dir
        
        if not search_dir.exists():
            return []
        
        return list(search_dir.rglob("*.json"))
    
    def get_latest_batch(self, model_id: str, task_type: str) -> Optional[BatchResult]:
        """Get the most recent batch result for a model/task."""
        results = self.list_results(model_id, task_type)
        batch_files = [f for f in results if f.name.startswith("batch_")]
        
        if not batch_files:
            return None
        
        # Sort by modification time, get latest
        latest = max(batch_files, key=lambda f: f.stat().st_mtime)
        return self.load_batch(latest)


def create_run_id() -> str:
    """Generate a unique run ID."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]
