# Enhancement Plan: Pipeline Features Phase 2

## Priority Order
1. **Enhancement Features** (this plan)
2. Integration Testing
3. E2E Testing
4. Documentation

---

## Enhancement 1: Additional Preprocessing Operators

### Goal
Expand preprocessing capabilities with commonly needed audio operations.

### New Operators to Add

| Operator | Description | ffmpeg filter | Params |
|----------|-------------|---------------|--------|
| `normalize_peak` | Peak normalization | `volume=replaygain=peak` | `target_db: float = -1.0` |
| `convert_samplerate` | Resample audio | `aresample` | `target_sr: int` (8000, 16000, 22050, 44100, 48000) |
| `mono_mix` | Stereo to mono downmix | `pan=mono\|c0=0.5*c0+0.5*c1` | none |
| `compress_dynamics` | Dynamic range compression | `acompressor` | `threshold_db: float = -20, ratio: float = 4` |
| `gate_noise` | Noise gate | `agate` | `threshold_db: float = -40` |

### Files to Modify

1. **`harness/pipeline_config.py`**
   - Add new entries to `PREPROCESSING_REGISTRY`
   - Update `PipelineConfig.to_ingest_config()` mappings

2. **`harness/media_ingest.py`**
   - Add fields to `IngestConfig` dataclass
   - Implement ffmpeg filter chains in `build_ffmpeg_filter_chain()`

3. **`client/src/pages/WorkbenchPage.tsx`**
   - UI already dynamically reads from `/api/pipelines/preprocessing` - no changes needed

### Implementation Estimate
~2-3 hours

---

## Enhancement 2: Pipeline Execution Progress Tracking

### Goal
Real-time step-by-step progress visibility in the UI during pipeline execution.

### Current State
- Manifest has `current_step` and `steps[].status`
- `/api/runs/{run_id}/status` returns this data
- UI polls but doesn't show granular step progress

### Proposed Changes

#### Backend

1. **`harness/session.py`** - Add progress percentage per step:
   ```python
   @dataclass
   class StepProgress:
       step_name: str
       status: str  # PENDING | RUNNING | COMPLETED | FAILED
       progress_pct: int  # 0-100
       message: Optional[str] = None
       started_at: Optional[str] = None
       estimated_remaining_s: Optional[int] = None
   ```

2. **Manifest schema update** (`manifest.json`):
   ```json
   {
     "steps": {
       "asr": {
         "status": "RUNNING",
         "progress_pct": 45,
         "progress_message": "Transcribing chunk 5/11",
         "started_at": "...",
         "estimated_remaining_s": 30
       }
     }
   }
   ```

3. **`server/api/runs.py`** - Enhance `/status` endpoint:
   ```python
   {
     "steps_progress": [
       {"name": "ingest", "status": "COMPLETED", "progress_pct": 100, "duration_ms": 1234},
       {"name": "asr", "status": "RUNNING", "progress_pct": 45, "message": "Transcribing..."},
       {"name": "diarization", "status": "PENDING", "progress_pct": 0}
     ]
   }
   ```

4. **Progress hooks in step implementations**:
   - `harness/asr/`: Report progress during transcription (chunk-based)
   - `harness/diarization/`: Report progress during speaker embedding

#### Frontend

1. **`client/src/pages/ResultsPage.tsx`** - Step progress visualization:
   - Vertical stepper with real-time status icons
   - Progress bars per step
   - ETA countdown for running step
   - Expandable step details (config used, duration)

2. **New component**: `PipelineProgress.tsx`
   ```tsx
   interface StepProgressProps {
     steps: Array<{
       name: string;
       status: 'pending' | 'running' | 'completed' | 'failed';
       progressPct: number;
       message?: string;
       durationMs?: number;
       estimatedRemainingS?: number;
     }>;
   }
   ```

### Implementation Estimate
~4-6 hours

---

## Enhancement 3: Pipeline Run History & Comparison

### Goal
Track and compare results across pipeline runs for the same input.

### Features

1. **Run History View**
   - List all runs grouped by input file hash
   - Show pipeline config diff between runs
   - Quick re-run with same/modified config

2. **Run Comparison**
   - Side-by-side transcript diff
   - Metrics comparison table (WER, speaker accuracy, timing)
   - Visual diff for diarization segments

### Data Model

#### Run Index Enhancement (`server/services/runs_index.py`)

```python
@dataclass
class RunIndexEntry:
    run_id: str
    input_hash: str  # SHA256 of source file
    input_filename: str
    pipeline_config_hash: str  # Hash of pipeline config for grouping
    created_at: str
    status: str
    steps_completed: List[str]
    # NEW
    preprocessing_ops: List[str]
    custom_steps: Optional[List[str]]
    template_used: Optional[str]
```

#### New Endpoints

1. **`GET /api/runs/by-input/{input_hash}`**
   - Returns all runs for a given input file
   - Sorted by created_at desc
   - Includes pipeline config summary

2. **`GET /api/runs/compare`**
   - Query params: `run_a={id}&run_b={id}`
   - Returns structured comparison:
     ```json
     {
       "runs": {"a": {...}, "b": {...}},
       "config_diff": {...},
       "metrics_comparison": {
         "transcript_word_count": {"a": 1234, "b": 1230, "diff": -4},
         "speaker_count": {"a": 3, "b": 3, "diff": 0}
       },
       "transcript_diff": {
         "added_words": 5,
         "removed_words": 9,
         "changed_segments": [...]
       }
     }
     ```

3. **`POST /api/runs/{run_id}/rerun`**
   - Re-run with optional config overrides
   - Links new run to original via `parent_run_id`

#### Frontend

1. **Run History Panel** (in ResultsPage):
   - Collapsible sidebar showing related runs
   - Quick config preview on hover
   - "Compare" button to select runs

2. **Comparison View** (`/compare?a={id}&b={id}`):
   - Split view with synced scrolling
   - Inline diff highlighting
   - Metrics delta badges

### Implementation Estimate
~6-8 hours

---

## Implementation Order

### Phase 2a: Quick Wins (Day 1)
1. ✅ Additional preprocessing operators
2. Basic step progress in manifest

### Phase 2b: Progress UI (Day 2)
1. Progress hooks in ASR/diarization steps
2. Frontend progress component
3. Real-time progress polling

### Phase 2c: History & Comparison (Day 3-4)
1. Run index enhancements
2. Comparison API endpoints
3. Frontend history panel
4. Comparison view

---

## Testing Strategy

### Unit Tests
- `tests/unit/test_preprocessing_ops.py` - New ffmpeg filter chains
- `tests/unit/test_progress_tracking.py` - Progress state transitions
- `tests/unit/test_run_comparison.py` - Comparison logic

### Integration Tests
- `tests/integration/test_pipeline_progress.py` - End-to-end progress flow
- `tests/integration/test_run_history.py` - History grouping

### E2E Tests
- `tests/e2e/test_progress_ui.sh` - Visual progress updates
- `tests/e2e/test_comparison_flow.sh` - Full comparison workflow

---

## Dependencies

- No new Python packages required
- ffmpeg filters already available
- Frontend uses existing component library

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Progress updates too frequent | Debounce manifest writes (max 1/sec) |
| Large transcript diffs slow UI | Paginate diff results, compute server-side |
| Run comparison edge cases | Validate both runs completed before comparing |

---

## Success Metrics

1. **Preprocessing**: All 5 new operators pass unit tests
2. **Progress**: UI shows step progress within 2s of manifest update
3. **History**: Runs grouped correctly by input hash
4. **Comparison**: Diff computed in <500ms for typical transcripts
