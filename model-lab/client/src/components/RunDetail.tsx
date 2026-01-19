import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { api } from '../lib/api';
import type { RunDetail as RunDetailType, ResultSummary } from '../lib/api';
import { Loader2, ArrowLeft, Trash2, Download } from 'lucide-react';
import { deriveProgressSignal, isStalled } from '../lib/runProgress';
import { getFailureStep } from '../lib/failureDetection';

interface RunDetailProps {
    onBack?: () => void;
}

// Pipeline definition: canonical order and labels (exported for helper functions)
export const PIPELINE_STEPS = [
    { key: 'ingest', label: 'Ingest audio' },
    { key: 'asr', label: 'Speech recognition' },
    { key: 'diarization', label: 'Speaker identification' },
    { key: 'alignment', label: 'Text alignment' },
    { key: 'chapters', label: 'Chapters' },
    { key: 'summarize_by_speaker', label: 'Summary' },
    { key: 'action_items_assignee', label: 'Action items' },
    { key: 'bundle', label: 'Bundle results' }
];

export default function RunDetail({ onBack }: RunDetailProps) {
    const { runId } = useParams<{ runId: string }>();
    const [status, setStatus] = useState<any>(null); // TODO: strict typing
    const [detail, setDetail] = useState<RunDetailType | null>(null);
    const [result, setResult] = useState<ResultSummary | null>(null);
    const [lastPolledAt, setLastPolledAt] = useState<number>(0);

    // Poll for status until terminal
    useEffect(() => {
        if (!runId) return;
        let pollTimer: ReturnType<typeof setInterval>;

        const checkStatus = async () => {
            try {
                const s = await api.getRunStatus(runId);
                setStatus(s);
                setLastPolledAt(Date.now());

                // Terminal State Handling
                if (['COMPLETED', 'FAILED', 'STALE'].includes(s.status)) {
                    if (pollTimer) clearInterval(pollTimer);

                    // Fetch Semantic Results (Metrics, Flags)
                    try {
                        const res = await api.getRunResults(runId);
                        setResult(res);

                        // If useful artifacts exist, load transcript
                        const shouldLoadArtifacts = s.status === 'COMPLETED' || (s.status === 'FAILED' && res.quality_flags.is_partial);

                        // Only load detail once
                        if (shouldLoadArtifacts) {
                            // Don't await here to avoid blocking status update
                            api.getTranscript(runId)
                                .then(data => setDetail(data))
                                .catch(err => console.error("Failed to load transcript", err));
                        }
                    } catch (err) {
                        console.error("Failed to load results", err);
                    }
                }
            } catch (e) {
                console.error("Failed to get status", e);
            }
        };

        checkStatus();
        pollTimer = setInterval(checkStatus, 2000);

        return () => clearInterval(pollTimer);
    }, [runId]);

    // Format Helpers
    const formatTime = (s: number) => {
        const mins = Math.floor(s / 60);
        const secs = Math.floor(s % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    const formatDuration = (seconds: number) => {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}m ${s}s`;
    };

    const relativeTime = (isoTimestamp: string) => {
        const diff = Date.now() - new Date(isoTimestamp).getTime();
        const seconds = Math.floor(diff / 1000);
        if (seconds < 60) return 'just now';
        const minutes = Math.floor(seconds / 60);
        if (minutes < 60) return `${minutes}m ago`;
        const hours = Math.floor(minutes / 60);
        return `${hours}h ago`;
    };

    // Extract audio duration with fallback (tiered: status → results → unknown)
    const getAudioDuration = (): number | null => {
        // Priority 1: From status (if manifest exposes it)
        // Priority 2: From results metrics
        if (result?.metrics?.audio_duration_s != null) {
            return result.metrics.audio_duration_s;
        }
        // Priority 3: Unknown
        return null;
    };

    // 1. Missing ID
    if (!runId) return <div className="p-8">Missing run id</div>;

    // 2. Initial Loading
    if (!status) return <div className="p-8 flex items-center gap-2"><Loader2 className="animate-spin" /> Loading run status...</div>;

    // Repeat run handler (frontend-only, passes context to workbench)
    const handleRepeatRun = () => {
        const params = new URLSearchParams({
            repeat_from: runId,
            file: status.input_filename || '',
        });
        window.location.href = `/lab/workbench?${params}`;
    };

    // 3. Queued / Running
    if (status.status === 'QUEUED' || status.status === 'RUNNING') {
        // Step descriptions (inline, not exported)
        const STEP_DESCRIPTIONS: Record<string, string> = {
            'ingest': 'Preparing audio file for processing',
            'asr': 'Transcribing spoken words',
            'diarization': 'Detecting who spoke when',
            'alignment': 'Synchronizing transcript with speakers',
            'chapters': 'Segmenting conversation topics',
            'summarize_by_speaker': 'Generating meeting summary',
            'action_items_assignee': 'Extracting tasks and decisions',
            'bundle': 'Packaging final outputs'
        };

        // Normalize current step (handle aliases)
        const normalizeStep = (s: string) => {
            const lower = s.toLowerCase();
            if (lower === 'transcription') return 'asr';
            if (lower === 'vad') return 'ingest';
            if (lower === 'metrics') return 'bundle';
            return lower;
        };

        const currentNormalized = status.current_step ? normalizeStep(status.current_step) : null;
        const completedNormalized = (status.steps_completed || []).map(normalizeStep);

        // Find current step index for causal completion logic
        const currentIndex = currentNormalized ? PIPELINE_STEPS.findIndex(s => s.key === currentNormalized) : -1;

        // Elapsed time since start
        let elapsed = '';
        if (status.started_at) {
            const start = new Date(status.started_at).getTime();
            const now = new Date().getTime();
            const sec = Math.floor((now - start) / 1000);
            if (sec > 0) {
                const m = Math.floor(sec / 60);
                const s = sec % 60;
                elapsed = `${m}m ${s}s`;
            }
        }

        // Derive progress signal from updated_at (centralized helper)
        const { secondsSinceProgress } = deriveProgressSignal(status.updated_at);
        const isStalledState = isStalled(status.status, secondsSinceProgress);
        const isRunningActive = status.status === 'RUNNING' && !isStalledState;
        const timeSinceProgress = secondsSinceProgress ?? 0; // fallback for display only

        // Progress signal messaging (user-friendly language)
        let progressMsg = '';
        let progressClass = 'text-gray-500';

        if (isStalledState) {
            // STALLED: No progress for beyond threshold
            const minutes = Math.floor(timeSinceProgress / 60);
            const seconds = timeSinceProgress % 60;
            progressMsg = `No progress signal for ${minutes}m ${seconds}s — may be stuck.`;
            progressClass = 'text-orange-600';
        } else if (isRunningActive) {
            // RUNNING (active): Show freshness
            if (timeSinceProgress < 10) {
                progressMsg = timeSinceProgress === 0
                    ? 'Processing is active. System just reported progress.'
                    : `Processing is active. Last progress signal ${timeSinceProgress}s ago.`;
                progressClass = 'text-green-600';
            } else if (timeSinceProgress < 30) {
                progressMsg = `Last progress signal ${timeSinceProgress}s ago.`;
                progressClass = 'text-gray-500';
            } else if (timeSinceProgress < 60) {
                progressMsg = `Last progress signal ${timeSinceProgress}s ago.`;
                progressClass = 'text-gray-600';
            } else {
                progressMsg = `Last progress signal ${timeSinceProgress}s ago.`;
                progressClass = 'text-gray-700';
            }
        }

        // Find current step details for description
        const currentStepInfo = PIPELINE_STEPS.find(s => s.key === currentNormalized);
        const currentStepDesc = currentNormalized ? STEP_DESCRIPTIONS[currentNormalized] : null;

        return (
            <div className="p-8 max-w-3xl mx-auto mt-12">
                {/* Header */}
                <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xl font-bold">
                        {status.status === 'QUEUED' ? 'Waiting in Queue' :
                            isStalled ? 'Run Stalled' : 'Processing Run'}
                    </h2>
                    {isStalledState ? (
                        <div className="px-3 py-1 bg-orange-100 text-orange-700 rounded text-sm font-semibold border border-orange-300">
                            STALLED
                        </div>
                    ) : (
                        <Loader2 className="animate-spin text-blue-600" size={28} />
                    )}
                </div>

                {/* Input Context Block */}
                <div className="bg-gray-50 border rounded-lg p-4 mb-6">
                    <h3 className="text-xs font-semibold text-gray-600 mb-2">Input</h3>
                    <div className="space-y-1">
                        <div className="flex items-center gap-2 text-sm">
                            <span>📄</span>
                            <span className="font-medium">{status.input_filename || runId}</span>
                        </div>
                        {(() => {
                            const audioDuration = getAudioDuration();
                            return audioDuration !== null ? (
                                <div className="flex items-center gap-2 text-sm text-gray-600">
                                    <span>⏱</span>
                                    <span>{formatDuration(audioDuration)} audio</span>
                                </div>
                            ) : (
                                <div className="flex items-center gap-2 text-sm text-gray-500">
                                    <span>⏱</span>
                                    <span>Duration: unknown</span>
                                </div>
                            );
                        })()}
                        {status.started_at && (
                            <div className="flex items-center gap-2 text-xs text-gray-500">
                                <span>⬆</span>
                                <span>Received {relativeTime(status.started_at)}</span>
                            </div>
                        )}
                    </div>
                </div>

                {/* Pipeline visualization */}
                <div className="bg-white border rounded-lg p-6 mb-6">
                    <h3 className="text-sm font-semibold text-gray-600 mb-3">Pipeline</h3>
                    <div className="space-y-2">
                        {PIPELINE_STEPS.map((step, idx) => {
                            // Step is completed if: explicitly in steps_completed OR before current_step
                            const isCompleted = completedNormalized.includes(step.key)
                                || (currentIndex >= 0 && idx < currentIndex);
                            const isCurrent = currentNormalized === step.key;
                            const isPending = !isCompleted && !isCurrent;

                            let icon = '○';
                            let textClass = 'text-gray-400';
                            if (isCompleted) {
                                icon = '✓';
                                textClass = 'text-green-600';
                            } else if (isCurrent) {
                                icon = '→';
                                textClass = 'text-blue-600 font-semibold';
                            }

                            return (
                                <div key={step.key} className={`flex items-start gap-2 ${textClass}`}>
                                    <span className="text-lg leading-none mt-0.5">{icon}</span>
                                    <span className="text-sm">{step.label}</span>
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* Current step details */}
                {currentStepInfo && (
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
                        <div className="text-sm font-semibold text-blue-900 mb-1">
                            Currently: {currentStepInfo.label}
                        </div>
                        <div className="text-sm text-blue-700">
                            {currentStepDesc}
                        </div>

                        {/* Scale-aware messaging for ASR */}
                        {currentStepInfo.key === 'asr' && (
                            <div className="text-xs text-blue-600 mt-3 border-t border-blue-200 pt-2 space-y-1">
                                {(() => {
                                    const audioDuration = getAudioDuration();
                                    if (audioDuration !== null && audioDuration <= 120) {
                                        return (
                                            <div>
                                                Short audio usually completes quickly, but may take longer depending on model and system load.
                                            </div>
                                        );
                                    } else if (audioDuration !== null && audioDuration > 120) {
                                        return (
                                            <div>
                                                Processing time scales with audio length. Longer files may take several minutes.
                                            </div>
                                        );
                                    }
                                    return null;
                                })()}
                                <div className="italic">
                                    Speech recognition is batch-processed, not real-time. Processing speed depends on model choice and system load.
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {/* Metadata */}
                <div className="bg-gray-50 rounded-lg p-4 space-y-2">
                    {elapsed && (
                        <div className="text-sm">
                            <span className="text-gray-600 font-medium">Time elapsed:</span>{' '}
                            <span className="font-mono text-gray-800">{elapsed}</span>
                        </div>
                    )}
                    {progressMsg && (
                        <div className={`text-sm font-medium ${progressClass}`}>
                            {progressMsg}
                        </div>
                    )}
                    <div className="text-xs text-gray-400 font-mono pt-2 border-t">
                        Run ID: {runId}
                    </div>
                </div>

                {/* Agency */}
                <div className="mt-6 text-sm text-gray-500 text-center">
                    You can leave this page safely. This run will stay highlighted in the list.
                </div>

                <div className="mt-4 text-center">
                    <button onClick={onBack} className="px-4 py-2 border rounded hover:bg-gray-50 text-sm text-gray-600">
                        Back to List
                    </button>
                </div>

                {/* Dev-only debug overlay */}
                {process.env.NODE_ENV === 'development' && (
                    <details className="mt-4 text-xs font-mono text-gray-500 bg-gray-100 p-3 rounded">
                        <summary className="cursor-pointer font-semibold">Debug (dev only)</summary>
                        <pre className="mt-2 overflow-x-auto">
                            {JSON.stringify({
                                run_id: runId,
                                status: status.status,
                                started_at: status.started_at,
                                updated_at: status.updated_at,
                                steps_completed: status.steps_completed,
                                current_step: status.current_step,
                                derived: {
                                    secondsSinceProgress,
                                    isStalledState,
                                },
                            }, null, 2)}
                        </pre>
                    </details>
                )}
            </div>
        );
    }

    // 4. Failed (check for partial results but still show failure UI)
    const isFailed = status.status === 'FAILED' || status.status === 'STALE';
    const hasPartialResults = isFailed && result?.quality_flags.is_partial;

    if (isFailed) {
        // Determine which step failed using centralized helper
        const { step: failedStep, isInferred, isUnknown } = getFailureStep(
            status.failure_step,
            status.steps_completed
        );

        return (
            <div className="p-8 max-w-3xl mx-auto mt-12">
                {/* Header */}
                <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xl font-bold text-red-700">Run Failed</h2>
                    <div className="px-3 py-1 bg-red-100 text-red-700 rounded text-sm font-semibold border border-red-300">
                        FAILED
                    </div>
                </div>

                {/* Error Message (always show for failed runs) */}
                {(status.error_message || isUnknown) && (
                    <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
                        {status.error_code && (
                            <div className="text-sm font-semibold text-red-900 mb-1">
                                {status.error_code}
                            </div>
                        )}
                        <div className="text-sm text-red-700">
                            {status.error_message || 'Run failed without detailed error message.'}
                        </div>
                    </div>
                )}

                {/* Pipeline visualization with failed step */}
                <div className="bg-white border rounded-lg p-6 mb-4">
                    <h3 className="text-sm font-semibold text-gray-600 mb-3">Pipeline</h3>
                    <div className="space-y-2">
                        {PIPELINE_STEPS.map((step) => {
                            const isCompleted = (status.steps_completed || []).includes(step.key);
                            const isFailed = step.key === failedStep;
                            const isPending = !isCompleted && !isFailed;

                            let icon = '⚪';
                            let textClass = 'text-gray-400';
                            if (isCompleted) {
                                icon = '✅';
                                textClass = 'text-green-600';
                            } else if (isFailed) {
                                icon = '❌';
                                textClass = 'text-red-600 font-semibold';
                            }

                            return (
                                <div key={step.key} className={`flex items-start gap-2 ${textClass}`}>
                                    <span className="text-lg leading-none mt-0.5">{icon}</span>
                                    <span className="text-sm">{step.label}</span>
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* Disclaimer for inferred failures */}
                {isInferred && (
                    <div className="text-sm text-gray-600 italic mb-6">
                        Failed step inferred from completed steps.
                    </div>
                )}

                {/* Explicit unknown state */}
                {isUnknown && (
                    <div className="text-sm text-gray-600 italic mb-6">
                        Failed (step unknown)
                    </div>
                )}

                {/* Metadata */}
                <div className="bg-gray-50 rounded-lg p-4 mb-6">
                    <div className="text-xs text-gray-400 font-mono">
                        Run ID: {runId}
                    </div>
                </div>

                {/* Actions */}
                <div className="flex gap-3 justify-center">
                    <button
                        onClick={handleRepeatRun}
                        className="px-4 py-2 flex items-center gap-2 border border-blue-600 text-blue-600 rounded hover:bg-blue-50"
                    >
                        🔁 Repeat run
                    </button>
                    <button onClick={onBack} className="px-4 py-2 border rounded hover:bg-gray-50">
                        Back to List
                    </button>
                </div>
            </div>
        );
    }

    // 5. Completed (or Partial)
    return (
        <div className="flex flex-col h-screen bg-gray-50">
            {/* Header */}
            <header className="bg-white border-b py-3 px-6 flex items-center justify-between shadow-sm z-10">
                <div className="flex items-center gap-4">
                    <button onClick={onBack} className="p-2 hover:bg-gray-100 rounded-full">
                        <ArrowLeft size={20} />
                    </button>
                    <div>
                        <div className="flex items-center gap-2">
                            <h2 className="font-bold text-lg">{runId}</h2>
                            {result?.quality_flags.is_partial && (
                                <span className="px-2 py-0.5 rounded text-xs font-semibold bg-orange-100 text-orange-700 border border-orange-200">
                                    PARTIAL
                                </span>
                            )}
                        </div>
                        <div className="flex items-center gap-4 text-xs text-gray-500 mt-1 font-mono">
                            {result?.metrics && (
                                <>
                                    {result.metrics.duration_s != null && (
                                        <span>⏱ {result.metrics.duration_s.toFixed(1)}s</span>
                                    )}
                                    {result.metrics.word_count != null && (
                                        <span>📝 {result.metrics.word_count} words</span>
                                    )}
                                    {result.metrics.confidence_avg != null && (
                                        <span>🎯 {(result.metrics.confidence_avg * 100).toFixed(1)}%</span>
                                    )}
                                </>
                            )}
                        </div>
                    </div>
                </div>

                {/* Actions */}
                <div className="flex gap-2">
                    <button
                        onClick={handleRepeatRun}
                        className="flex items-center gap-2 px-3 py-1.5 text-sm border border-blue-600 text-blue-600 rounded hover:bg-blue-50"
                        title="Start a new run with the same input file"
                    >
                        🔁 Repeat run
                    </button>
                    <a href={api.getMeetingPackZipUrl(runId)} target="_blank" rel="noreferrer" className="flex items-center gap-2 px-3 py-1.5 text-sm bg-blue-600 text-white rounded hover:bg-blue-700">
                        <Download size={16} /> Meeting Pack ZIP
                    </a>
                </div>
            </header>

            {/* Content Scroller */}
            <div className="flex-1 overflow-y-auto p-8">
                <div className="max-w-3xl mx-auto bg-white rounded-lg shadow-sm border p-8 min-h-[50vh]">
                    {!detail ? (
                        <div className="flex items-center gap-2 text-gray-500">
                            <Loader2 className="animate-spin" size={20} /> Loading transcript...
                        </div>
                    ) : (
                        <div className="space-y-6">
                            {(detail.segments || []).length === 0 && (
                                <div className="text-gray-400 text-center italic">No transcript segments found.</div>
                            )}

                            {detail.segments.map((seg, idx) => (
                                <div key={idx} className="flex gap-4 group">
                                    {/* Timestamp */}
                                    <div className="w-16 flex-shrink-0 text-xs text-gray-400 font-mono mt-1 select-none">
                                        {seg.start_s != null ? formatTime(seg.start_s) : '--:--'}
                                    </div>

                                    {/* Content */}
                                    <div className="flex-1">
                                        {seg.speaker && <div className="font-bold text-xs text-gray-600 mb-0.5">{seg.speaker}</div>}
                                        <p className="text-gray-800 leading-relaxed text-lg">
                                            {seg.text}
                                        </p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
