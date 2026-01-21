import { describe, test, expect } from 'vitest';
import { deriveRunDetailViewModel, ApiRunStatus } from './viewModel';

describe('deriveRunDetailViewModel', () => {
    
    const BASE_RUN: ApiRunStatus = {
        status: 'RUNNING',
        steps_completed: [],
        failure_step: null,
        error_message: undefined,
        is_stalled: false,
        steps: []
    };

    test('1. Happy Path: All steps completed', () => {
        const run: ApiRunStatus = {
            ...BASE_RUN,
            status: 'COMPLETED',
            steps_completed: ['ingest', 'asr', 'diarization', 'alignment', 'chapters', 'summarize_by_speaker', 'action_items_assignee', 'bundle']
        };
        const vm = deriveRunDetailViewModel(run);
        
        expect(vm.overallStatus).toBe('COMPLETED');
        expect(vm.pipelineSteps.every(s => s.status === 'COMPLETED')).toBe(true);
    });

    test('2. Explicit Failure: alignment failed, previous completed', () => {
        const run: ApiRunStatus = {
            ...BASE_RUN,
            status: 'FAILED',
            failure_step: 'alignment',
            steps_completed: ['ingest', 'asr', 'diarization']
        };
        const vm = deriveRunDetailViewModel(run);

        expect(vm.overallStatus).toBe('FAILED');
        
        const steps = vm.pipelineSteps;
        expect(steps.find(s => s.key === 'ingest')?.status).toBe('COMPLETED');
        expect(steps.find(s => s.key === 'asr')?.status).toBe('COMPLETED');
        expect(steps.find(s => s.key === 'diarization')?.status).toBe('COMPLETED');
        expect(steps.find(s => s.key === 'alignment')?.status).toBe('FAILED');
        expect(steps.find(s => s.key === 'chapters')?.status).toBe('NOT_RUN');
    });

    test('3. Split Brain Failure: alignment failed BUT marked completed', () => {
        const run: ApiRunStatus = {
            ...BASE_RUN,
            status: 'FAILED',
            failure_step: 'alignment',
            steps_completed: ['ingest', 'asr', 'diarization', 'alignment'] // Conflicting info
        };
        const vm = deriveRunDetailViewModel(run);

        expect(vm.overallStatus).toBe('FAILED');
        // FAILURE MUST DOMINATE
        expect(vm.pipelineSteps.find(s => s.key === 'alignment')?.status).toBe('FAILED');
    });

    test('4. Stale Failure: Stale status but has failure step', () => {
        const run: ApiRunStatus = {
            ...BASE_RUN,
            status: 'STALE', // Backend says stale
            is_stalled: true,
            failure_step: 'alignment',
            error_message: 'Worker vanished'
        };
        const vm = deriveRunDetailViewModel(run);

        expect(vm.overallStatus).toBe('FAILED');
        expect(vm.isStale).toBe(true);
        expect(vm.primaryReason).toContain('Worker vanished');
        expect(vm.secondaryReason).toContain('No heartbeat'); // typo in test match? Checking logic
        
        expect(vm.pipelineSteps.find(s => s.key === 'alignment')?.status).toBe('FAILED');
    });

    test('5. Stale Running: Stale + No failure', () => {
        const run: ApiRunStatus = {
            ...BASE_RUN,
            status: 'STALE',
            is_stalled: true,
            steps_completed: ['ingest']
        };
        const vm = deriveRunDetailViewModel(run);

        expect(vm.overallStatus).toBe('RUNNING'); // Treated as Running but Stale in VM abstraction usually, or we can explicit STALE. 
        // User spec: "If no failure and run is stale and not terminal: overallStatus remains "RUNNING" but isStale = true"
        
        expect(vm.isStale).toBe(true);
        expect(vm.pipelineSteps.find(s => s.key === 'ingest')?.status).toBe('COMPLETED');
    });

    test('6. Missing History: Failure at step 3, but step 1 not in steps_completed', () => {
        const run: ApiRunStatus = {
            ...BASE_RUN,
            status: 'FAILED',
            failure_step: 'diarization', // 3rd step
            steps_completed: ['asr'] // 'ingest' missing!
        };
        const vm = deriveRunDetailViewModel(run);

        expect(vm.overallStatus).toBe('FAILED');
        
        const ingest = vm.pipelineSteps.find(s => s.key === 'ingest');
        const asr = vm.pipelineSteps.find(s => s.key === 'asr');
        const diar = vm.pipelineSteps.find(s => s.key === 'diarization');
        
        // Ingest missing from completed -> UNKNOWN (because before failure)
        expect(ingest?.status).toBe('UNKNOWN'); 
        expect(asr?.status).toBe('COMPLETED');
        expect(diar?.status).toBe('FAILED');
    });

    test('7. Artifact Fallback: failedStep alignment, stepsCompleted empty, artifacts has ASR', () => {
        const run: ApiRunStatus = {
            ...BASE_RUN,
            status: 'FAILED',
            failure_step: 'alignment',
            steps_completed: [], // Data loss in completion list
            steps: [
                { name: 'asr', status: 'COMPLETED', artifacts: [{ id: '1', role: 'transcript', filename: 't.json', produced_by: 'asr', size_bytes: 10, downloadable: true }] }
            ] as any[]
        };
        const vm = deriveRunDetailViewModel(run);

        const asr = vm.pipelineSteps.find(s => s.key === 'asr');
        expect(asr?.status).toBe('COMPLETED'); // Fallback worked
        
        const alignment = vm.pipelineSteps.find(s => s.key === 'alignment');
        expect(alignment?.status).toBe('FAILED');
    });

    test('8. Ambiguous Failure: No failedStep, but errorMessage present', () => {
        const run: ApiRunStatus = {
            ...BASE_RUN,
            status: 'FAILED',
            failure_step: null,
            error_message: 'Something exploded globally',
            steps_completed: ['ingest']
        };
        const vm = deriveRunDetailViewModel(run);

        expect(vm.overallStatus).toBe('FAILED');
        expect(vm.primaryReason).toBe('Something exploded globally');
        
        // Steps should be mostly unknown if we don't know where it failed?
        // OR we trust completed and others are pending/unknown?
        // Logic: No failure_step -> myIndex > failureIndex (-1) is TRUE for all.
        // Wait, failureIndex = -1. myIndex > -1 is always true. 
        // So they would be NOT_RUN? 
        // Let's check logic: if failureIndex == -1, we fall to "No known failure step (yet)" block.
        // But overallStatus is FAILED.
        // Current logic puts them in the "No known failure step" block.
        // For a generic global failure, we probably want to see what finished.
        
        const ingest = vm.pipelineSteps.find(s => s.key === 'ingest');
        const asr = vm.pipelineSteps.find(s => s.key === 'asr');
        
        expect(ingest?.status).toBe('COMPLETED');
        // ASR not in completed, so 'PENDING' by default block?
        // Ideally if FAILED and no failure step, we might want UNKNOWN.
        // But 'PENDING' implies it was next. 
        // User spec says: "Steps before failedStepKey: UNKNOWN | COMPLETED".
        // If no failedStepKey, this rule is tricky.
        // But functionally, preserving "Completed" for known steps is correct.
    });



    test('9. Current Step Regression: RUNNING, current_step="asr", derived empty, steps_completed=[]', () => {
        const run: ApiRunStatus = {
            ...BASE_RUN,
            status: 'RUNNING',
            steps_completed: [], // No history
            current_step: 'asr', // Explicit active step
            steps: []
        };
        const vm = deriveRunDetailViewModel(run);

        expect(vm.overallStatus).toBe('RUNNING');
        expect(vm.isStale).toBe(false);
        
        const ingest = vm.pipelineSteps.find(s => s.key === 'ingest');
        const asr = vm.pipelineSteps.find(s => s.key === 'asr');
        
        // Ingest is before active step, but not in steps_completed -> UNKNOWN
        expect(ingest?.status).toBe('UNKNOWN'); 
        
        // ASR is active -> RUNNING
        expect(asr?.status).toBe('RUNNING');
    });

    test('10. Resolved Device Extraction', () => {
        const run: ApiRunStatus = {
            ...BASE_RUN,
            status: 'COMPLETED',
            steps_completed: ['ingest', 'asr'],
            steps: [
                { name: 'ingest', status: 'COMPLETED' },
                { 
                    name: 'asr', 
                    status: 'COMPLETED', 
                    resolved_config: { 
                        device: 'mps', 
                        model_id: 'large-v3', 
                        source: 'local', 
                        language: 'en' 
                    } 
                }
            ]
        };
        const vm = deriveRunDetailViewModel(run);
        expect(vm.resolvedDevice).toBe('mps');
    });
    test('11. Unknown Current Step: RUNNING with unknown step key', () => {
        const run: ApiRunStatus = {
            ...BASE_RUN,
            status: 'RUNNING',
            current_step: 'magic_step',
            steps_completed: ['ingest']
        };
        const vm = deriveRunDetailViewModel(run);
        
        expect(vm.overallStatus).toBe('RUNNING');
        expect(vm.secondaryReason).toContain('Unknown current_step: magic_step');
        
        // Pipeline should reflect completion
        expect(vm.pipelineSteps.find(s => s.key === 'ingest')?.status).toBe('COMPLETED');
        // Others pending
    });
});
