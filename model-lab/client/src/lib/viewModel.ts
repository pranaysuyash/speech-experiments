import type { Step } from './api';

// Re-export types if needed or define them here to avoid circular dependency
//Ideally PIPELINE_STEPS should be in a shared constant file, but importing from RunDetail is fine for now if RunDetail exports it.
// To avoid circular dependency issues if RunDetail imports this, we might need to duplicate or move PIPELINE_STEPS. 
// For now, let's assume we can move PIPELINE_STEPS to a shared location or just define the keys here to be safe and independent.

const ORDERED_PIPELINE_KEYS = [
    'ingest',
    'asr',
    'diarization',
    'alignment',
    'chapters',
    'summarize_by_speaker',
    'action_items_assignee',
    'bundle'
];

const STEP_LABELS: Record<string, string> = {
    'ingest': 'Ingest audio',
    'asr': 'Speech recognition',
    'diarization': 'Speaker identification',
    'alignment': 'Text alignment',
    'chapters': 'Chapters',
    'summarize_by_speaker': 'Summary',
    'action_items_assignee': 'Action items',
    'bundle': 'Bundle results'
};

export type StepStatus = "COMPLETED" | "FAILED" | "RUNNING" | "PENDING" | "NOT_RUN" | "UNKNOWN";
export type OverallStatus = "FAILED" | "RUNNING" | "COMPLETED" | "QUEUED" | "UNKNOWN" | "STALE"; // Keeping STALE as a status for backward compat but VM prefers FAILED/RUNNING + isStale

export interface ViewModel {
    overallStatus: "FAILED" | "RUNNING" | "COMPLETED" | "QUEUED" | "UNKNOWN";
    isStale: boolean;
    primaryReason: string | null;
    secondaryReason: string | null;
    pipelineSteps: Array<{ key: string; label: string; status: StepStatus }>;
    resolvedDevice?: string; // e.g. "cpu", "cuda", "mps"
}

export interface ArtifactSimple {
    role: string;
}

export interface ApiRunStatus {
    status: string;
    steps_completed: string[];
    failure_step?: string | null;
    error_message?: string;
    updated_at?: string;
    steps?: Step[]; // For artifacts
    heartbeat_age_s?: number; // Depending on what the API actually returns, often calculated client side or present
    // Helper to accept the raw status object we saw in RunDetail
    is_stalled?: boolean;
    current_step?: string | null;
}

// Helper to normalize step names to pipeline keys
function normalizeStepKey(stepName: string | null | undefined): string | null {
    if (!stepName) return null;
    const lower = stepName.toLowerCase();
    // Map known aliases
    if (lower === 'transcription') return 'asr';
    if (lower === 'vad') return 'ingest'; // often mapped
    if (lower === 'metrics') return 'bundle';
    // Return if it matches a valid key, otherwise return the lower key (or null check later)
    return lower;
}

export function deriveRunDetailViewModel(run: ApiRunStatus): ViewModel {
    // 1. Determine Overall Status & Staleness
    // "Stale" is a liveness property. "Failed" is a terminal outcome.
    // Logic: If explicitly FAILED (via status or error), it is FAILED.
    // If explicitly COMPLETED/CANCELLED, it is that.
    // If RUNNING/QUEUED, checking staleness matters more for the "Status" label.
    
    // However, the VM contract says:
    // overallStatus: FAILED | RUNNING | COMPLETED | QUEUED | UNKNOWN
    // isStale: boolean

    const isStale = run.is_stalled || (run.status === 'STALE'); // status 'STALE' is often returned by backend
    let overallStatus: ViewModel['overallStatus'] = 'UNKNOWN';

    // Map backend status to VM status strict set
    const s = run.status;
    if (s === 'FAILED' || s === 'CANCELLED' || run.failure_step || run.error_message) {
        overallStatus = 'FAILED';
    } else if (s === 'COMPLETED') {
        overallStatus = 'COMPLETED';
    } else if (s === 'RUNNING' || s === 'STALE') {
        // STALE in backend means "Running but no heartbeat". 
        // In VM we call it RUNNING (with isStale=true) UNLESS we have failure evidence.
        overallStatus = 'RUNNING';
    } else if (s === 'QUEUED') {
        overallStatus = 'QUEUED';
    }

    // Double check invariants: If failed_step or error_message exists, it IS failed.
    if (run.failure_step || run.error_message) {
        overallStatus = 'FAILED';
    }

    // 2. Reasons
    let primaryReason: string | null = null;
    let secondaryReason: string | null = null;

    if (overallStatus === 'FAILED') {
        primaryReason = run.error_message || (run.failure_step ? `Failed at ${run.failure_step}` : "Run failed (unknown error)");
        if (isStale) {
            secondaryReason = "Worker stopped responding (No heartbeat)";
        }
    } else if (isStale) {
        // If simply stale but not failed
        primaryReason = "Worker stopped responding";
        secondaryReason = "No heartbeat recently";
    }

    // 3. Pipeline Steps
    const pipelineSteps: Array<{ key: string; label: string; status: StepStatus }> = [];

    // Resolve Active Step from current_step
    const rawCurrentStep = run.current_step;
    const activeStepKey = normalizeStepKey(run.current_step);
    
    // If we have an active step that isn't in our valid list, we might want to note it (or just ignore it).
    // The user asked to surface "Unknown current_step" as a secondary reason if invalid? 
    // "If activeKey is not in your ordered pipeline list, set activeKey = null and surface secondaryReason"
    let validatedActiveKey: string | null = null;
    if (activeStepKey && ORDERED_PIPELINE_KEYS.includes(activeStepKey)) {
        validatedActiveKey = activeStepKey;
    } else if (activeStepKey) {
        // It exists but is not in our known list (e.g. unknown system step).
        if (overallStatus === 'RUNNING' && !secondaryReason) {
             secondaryReason = `Unknown current_step: ${rawCurrentStep}`;
        }
    }

    for (let i = 0; i < ORDERED_PIPELINE_KEYS.length; i++) {
        const key = ORDERED_PIPELINE_KEYS[i];
        // Default
        let status: StepStatus = 'UNKNOWN';

        // Evidence
        const inCompletedList = (run.steps_completed || []).includes(key);
        
        // Artifact fallback: Check if any artifact has a role mapping to this step?
        // Or simply if the step in run.steps has artifacts.
        // run.steps is Array of {name, status, artifacts}.
        // We find the specific step entry in the list
        const stepDetail = run.steps?.find(s => s.name === key);
        const hasArtifacts = (stepDetail?.artifacts && stepDetail.artifacts.length > 0);
        
        // Invariant: Failed Step
        const isTheFailedStep = (run.failure_step === key);
        
        // Invariant: Future Steps
        // If we have a failure step, anything AFTER it is NOT_RUN (unless it somehow has artifacts? No, trust failure location).
        const failureIndex = run.failure_step ? ORDERED_PIPELINE_KEYS.indexOf(run.failure_step) : -1;
        const myIndex = i; // Current index
        
        if (isTheFailedStep) {
            status = 'FAILED';
        } else if (failureIndex !== -1 && myIndex > failureIndex) {
            status = 'NOT_RUN';
        } else if (failureIndex !== -1 && myIndex < failureIndex) {
            // Steps BEFORE failure
            // Contract: COMPLETED only if proven by stepsCompleted OR artifacts.
            // Otherwise UNKNOWN.
            if (inCompletedList || hasArtifacts) {
                status = 'COMPLETED';
            } else {
                status = 'UNKNOWN';
            }
        } else {
            // No known failure step (yet).
            
            // Priority 1: Is this the specific Active Step?
            // "Active step: RUNNING"
            if (overallStatus === 'RUNNING' && key === validatedActiveKey) {
                status = 'RUNNING';
            } else if (validatedActiveKey && myIndex > ORDERED_PIPELINE_KEYS.indexOf(validatedActiveKey)) {
                // If we have a valid active key, anything after it is PENDING
                 status = 'PENDING';
            } else if (validatedActiveKey && myIndex < ORDERED_PIPELINE_KEYS.indexOf(validatedActiveKey)) {
                // Steps BEFORE active: UNKNOWN (unless artifacts prove them)
                if (inCompletedList || hasArtifacts) {
                    status = 'COMPLETED';
                } else {
                    status = 'UNKNOWN';
                }
            } else {
                // Fallback (No valid current_step) - Use original inference logic or strict evidence?
                // User said: "Prefer run.current_step... Only fall back to derived if run.current_step is missing."
                // Original inference logic being:
                if (inCompletedList) {
                    status = 'COMPLETED';
                } else if (overallStatus === 'COMPLETED') {
                     status = hasArtifacts ? 'COMPLETED' : 'UNKNOWN';
                } else if (overallStatus === 'RUNNING') {
                     // Fallback inference: First non-completed is RUNNING? 
                     // User's strict mode implies we probably shouldn't guess too wildly, 
                     // but legacy behavior for old runs might need it.
                     // Let's keep the caching logic but simpler.
                     status = 'PENDING'; 
                } else {
                    status = 'PENDING'; // Default for queued etc
                }
            }
        }
        
        // Refinement for Running/current (Fallback if no validatedActiveKey)
        // If we are strictly RUNNING, and this step is NOT completed, and it is the first non-completed step?
        if (overallStatus === 'RUNNING' && status === 'PENDING' && !validatedActiveKey) {
             // Is this the current step?
             // Since we iterate in order, if all previous were completed, this one is likely Running.
             const prevIndex = myIndex - 1;
             const prevCompleted = prevIndex < 0 || pipelineSteps[prevIndex].status === 'COMPLETED';
             
             if (prevCompleted && !run.failure_step) {
                 status = 'RUNNING';
             }
        }

        pipelineSteps.push({
            key,
            label: STEP_LABELS[key] || key,
            status
        });
    }

    // Extract resolved device if available (from ASR step usually)
    const asrStep = run.steps?.find(s => s.name === 'asr');
    const resolvedDevice = asrStep?.resolved_config?.device;

    return {
        overallStatus,
        isStale: !!isStale,
        primaryReason,
        secondaryReason,
        pipelineSteps,
        resolvedDevice
    };
}
