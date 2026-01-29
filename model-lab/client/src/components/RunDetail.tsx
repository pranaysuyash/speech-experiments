import { useEffect, useState, useCallback, useMemo, useRef } from 'react';
import { useParams } from 'react-router-dom';
import { api } from '../lib/api';
import type { RunDetail as RunDetailType, ResultSummary, StepProgress as ApiStepProgress } from '../lib/api';
import { Loader2, ArrowLeft, Download } from 'lucide-react';
// import { deriveProgressSignal, isStalled } from '../lib/runProgress'; // Removed client-side heuristics
import { deriveRunDetailViewModel } from '../lib/viewModel';
import { PipelineProgress, type StepProgressData } from './PipelineProgress';
import RunHistory from './RunHistory';
import { RunEventStream, type RunEvent } from '../lib/ws';

interface RunDetailProps {
    onBack?: () => void;
}

import { DebugPanel } from './DebugPanel';

function convertStepsProgress(apiSteps?: ApiStepProgress[]): StepProgressData[] {
    if (!apiSteps) return [];
    return apiSteps.map(step => ({
        name: step.name,
        status: step.status.toLowerCase() as StepProgressData['status'],
        progressPct: step.progress_pct,
        message: step.message,
        durationMs: step.duration_ms,
        estimatedRemainingS: step.estimated_remaining_s,
        startedAt: step.started_at,
        endedAt: step.ended_at,
    }));
}

// Helper for actionable error messages
function getActionableError(step: string | null, msg: string | null): { title: string, action: string } | null {
    if (!step && !msg) return null;

    if (msg?.includes("E_ARTIFACT_REGISTRY_MISSING")) {
        return {
            title: "Artifact Registry Error (Schema v2)",
            action: "Critical Data Gap: A required artifact is missing from the registry. This usually means a step failed to return the expected normalized output. Rerun the step or check the backend norms."
        };
    }

    if (msg?.includes("Simulated failure")) {
        return {
            title: "Simulated Test Failure",
            action: "This failure was injected for testing purposes. Reset the environment or check MODEL_LAB_FAIL_STEP."
        };
    }

    if (step === "ingest") {
        return {
            title: "Ingest Failed",
            action: "Check input file format and FFmpeg availability."
        };
    }

    return {
        title: `Step '${step}' Failed`,
        action: "Review logs for stack trace. Attempt retry if transient."
    };
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

    const vm = useMemo(() => {
        if (!status) return null;
        return deriveRunDetailViewModel(status);
    }, [status]);

    // Guard: runId is required
    if (!runId) {
        return <div className="p-8 text-red-600">Error: Run ID is required</div>;
    }

    const handleKill = useCallback(async () => {
        if (!confirm("Are you sure you want to stop this run?")) return;
        try {
            await api.killRun(runId);
        } catch (err) {
            alert("Failed to kill run");
            console.error(err);
        }
    }, [runId]);

    const handleRetry = useCallback(async () => {
        try {
            await api.retryRun(runId);
        } catch (err) {
            alert("Failed to retry run");
            console.error(err);
        }
    }, [runId]);

    // Repeat run handler (frontend-only, passes context to workbench)
    const handleRepeatRun = useCallback(() => {
        const params = new URLSearchParams();
        params.set('repeat_from', runId);
        params.set('file', status?.input_metadata?.filename || status?.input_filename || '');
        window.location.href = `/lab/workbench?${params}`;
    }, [runId, status]);

    // Unified Actions Renderer
    const renderActions = (className = "flex items-center gap-2") => {
        const isRunning = vm ? (vm.overallStatus === 'RUNNING' || vm.overallStatus === 'QUEUED') : false;
        const isFailed = vm ? (vm.overallStatus === 'FAILED') : false;
        // Only show meeting pack if we have a result or useful artifacts
        // For partial failures, we might have artifacts.
        const showArtifacts = result || (isFailed && status.steps_completed?.length > 0);
        // Actually api.getMeetingPackZipUrl is always valid URL, but might 404 if no bundle.
        // We rely on 'result' existence or specific status.

        return (
            <div className={className}>
                {/* Back is context dependent, but usually present. In Header it is separate. 
                    In Running/Failed views it is part of the block. 
                    Let's allow onBack prop or assume this replaces the block which included Back?
                    The running/failed blocks had "Back" button. The Success header has Back separately.
                    Refactor: Header handles Back separately. renderActions handles the Rest.
                */}

                {isRunning && (
                    <button onClick={handleKill} className="px-3 py-1.5 border border-red-200 bg-red-50 rounded hover:bg-red-100 text-sm text-red-600 font-semibold flex items-center gap-2">
                        🛑 Stop
                    </button>
                )}

                {isFailed && (
                    <button onClick={handleRetry} className="px-3 py-1.5 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm font-semibold flex items-center gap-2">
                        🔄 Retry
                    </button>
                )}

                <button
                    onClick={handleRepeatRun}
                    className="px-3 py-1.5 border border-blue-600 text-blue-600 rounded hover:bg-blue-50 text-sm flex items-center gap-2 bg-white"
                    title="Start a new run with the same input file"
                >
                    🔁 Repeat
                </button>

                {showArtifacts && (
                    <a
                        href={api.getMeetingPackZipUrl(runId)}
                        target="_blank"
                        rel="noreferrer"
                        className="flex items-center gap-2 px-3 py-1.5 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
                    >
                        <Download size={16} /> Pack
                    </a>
                )}
            </div>
        );
    };

    // WebSocket stream ref for real-time events
    const wsRef = useRef<RunEventStream | null>(null);

    // Poll for status until terminal, with WebSocket streaming for real-time updates
    useEffect(() => {
        if (!runId) return;
        let pollTimer: ReturnType<typeof setInterval>;
        let isActive = true;

        const loadTranscript = () => {
            api.getTranscript(runId)
                .then(data => {
                    if (!isActive) return;
                    setDetail(data);
                })
                .catch(err => {
                    console.error("Failed to load transcript", err);
                    if (!isActive) return;
                    setDetail({
                        run_id: runId,
                        segments: [],
                        chapters: []
                    });
                });
        };

        const checkStatus = async () => {
            try {
                const s = await api.getRunStatus(runId);
                setStatus(s);

                // Terminal State Handling
                if (['COMPLETED', 'FAILED', 'STALE', 'CANCELLED'].includes(s.status)) {
                    if (pollTimer) clearInterval(pollTimer);

                    // Close WebSocket when run is terminal
                    if (wsRef.current) {
                        wsRef.current.close();
                        wsRef.current = null;
                    }

                    // Fetch Semantic Results (Metrics, Flags)
                    try {
                        const res = await api.getRunResults(runId);
                        setResult(res);

                        // Trigger transcript load regardless of artifact list (backend will return empty if none)
                        loadTranscript();
                    } catch (err) {
                        console.error("Failed to load results", err);
                    }
                }
            } catch (e) {
                console.error("Failed to get status", e);
            }
        };

        // Initialize WebSocket for real-time events
        const initWebSocket = () => {
            if (wsRef.current) return; // Already connected

            // Create stream (connection starts automatically)
            const ws = new RunEventStream(runId);
            wsRef.current = ws;

            ws.onEvent((event: RunEvent) => {
                if (!isActive) return;

                // Handle different event types
                switch (event.type) {
                    case 'step_started':
                    case 'step_completed':
                    case 'step_failed':
                    case 'step_progress':
                        // Real-time step updates - trigger a status refresh
                        // This is more responsive than waiting for the next poll
                        checkStatus();
                        break;

                    case 'run_completed':
                    case 'run_failed':
                    case 'run_cancelled':
                        // Terminal event - fetch final status
                        checkStatus();
                        break;

                    case 'heartbeat':
                        // Heartbeat confirms connection is alive
                        break;

                    default:
                        // Log unknown events for debugging
                        console.debug('Unknown WS event:', event);
                }
            });

            ws.onError((_error) => {
                console.warn('WebSocket error, falling back to polling');
                // WebSocket failed, but polling will continue as backup
            });

            // Connect after setting up handlers
            ws.connect();
        };

        checkStatus();
        pollTimer = setInterval(checkStatus, 2000);

        // Start WebSocket connection for real-time updates
        initWebSocket();

        return () => {
            isActive = false;
            if (pollTimer) clearInterval(pollTimer);
            if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
            }
        };
    }, [runId]);

    // Format Helpers
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
        // Priority 0: status input metadata (available during RUNNING once ingest computed it)
        if (status?.input_metadata?.duration_s != null) return status.input_metadata.duration_s;

        // Priority 1: results metrics (terminal states)
        if (result?.metrics?.audio_duration_s != null) return result.metrics.audio_duration_s;

        return null;
    };

    const isMeaningful = (v?: string | null) => {
        if (!v) return false;
        const s = String(v).trim().toLowerCase();
        return s !== '' && s !== 'unknown' && s !== 'n/a' && s !== 'null' && s !== 'undefined';
    };

    const failedSteps = (status?.steps || []).filter((s: any) => s.status === 'FAILED');
    const hasStepFailures = failedSteps.length > 0;

    const renderStepFailures = () => {
        if (!hasStepFailures) return null;
        return (
            <div className="bg-red-50 border border-red-200 rounded-lg p-6 mb-6">
                <h3 className="text-sm font-semibold text-red-900 mb-3">Step Failures</h3>
                <div className="space-y-3">
                    {failedSteps.map((step: any) => (
                        <div key={step.name} className="bg-white border border-red-300 rounded p-3">
                            <div className="font-semibold text-red-800 text-sm mb-1">❌ {step.name}</div>
                            {step.error && (
                                <>
                                    <div className="flex items-center gap-2 mb-1">
                                        <span className="text-xs text-red-700 font-mono">{step.error.code || step.error.type}</span>
                                        {step.error.recoverable && (
                                            <span className="text-xs bg-yellow-100 text-yellow-800 px-1.5 py-0.5 rounded">
                                                Recoverable
                                            </span>
                                        )}
                                    </div>
                                    <div className="text-sm text-red-600">{step.error.message}</div>
                                    {step.error.traceback_path && (
                                        <div className="text-xs text-gray-500 mt-1">
                                            Traceback: {step.error.traceback_path}
                                        </div>
                                    )}
                                </>
                            )}
                            {step.duration_ms && (
                                <div className="text-xs text-gray-500 mt-2">
                                    Failed after {(step.duration_ms / 1000).toFixed(1)}s
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>
        );
    };

    const renderArtifacts = () => {
        const allArtifacts: Array<{ stepName: string; artifact: any }> = [];
        status?.steps?.forEach((step: any) => {
            if (step.artifacts && step.status === 'COMPLETED') {
                step.artifacts.forEach((art: any) => allArtifacts.push({ stepName: step.name, artifact: art }));
            }
        });
        if (allArtifacts.length === 0) return null;

        const formatBytes = (bytes?: number) => {
            if (bytes == null) return '?';
            if (bytes < 1024) return `${bytes} B`;
            if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
            return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
        };

        return (
            <div className="bg-white border rounded-lg p-6 mb-6">
                <h3 className="text-sm font-semibold text-gray-600 mb-3">Artifacts</h3>
                <div className="space-y-2">
                    {allArtifacts.map(({ artifact }) => (
                        <div key={artifact.id} className="flex items-center justify-between border rounded p-3 bg-gray-50">
                            <div className="flex items-center gap-3">
                                <span className="text-gray-400">📄</span>
                                <div>
                                    <div className="font-medium text-sm text-gray-800">
                                        {artifact.filename || artifact.id || 'unknown'}
                                    </div>
                                    <div className="text-xs text-gray-500">
                                        {artifact.role || 'unknown'} · {formatBytes(artifact.size_bytes)}
                                    </div>
                                </div>
                            </div>
                            <button
                                disabled={!artifact.downloadable}
                                onClick={() => artifact.downloadable && window.open(`/api/runs/${runId}/artifacts/${artifact.id}`, '_blank')}
                                className={`text-xs px-3 py-1 border rounded ${artifact.downloadable
                                    ? 'bg-blue-50 text-blue-600 border-blue-200 hover:bg-blue-100 cursor-pointer'
                                    : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                                    }`}
                            >
                                Download
                            </button>
                        </div>
                    ))}
                </div>
            </div>
        );
    };

    // 1. Missing ID
    if (!runId) return <div className="p-8">Missing run id</div>;

    // 2. Initial Loading
    if (!status) return <div className="p-8 flex items-center gap-2"><Loader2 className="animate-spin" /> Loading run status...</div>;

    // 3. Queued / Running
    if (vm?.overallStatus === 'QUEUED' || vm?.overallStatus === 'RUNNING') {
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

        const currentStepVM = vm.pipelineSteps.find(s => s.status === 'RUNNING') || vm.pipelineSteps.find(s => s.status === 'PENDING');
        const currentNormalized = currentStepVM?.key || null;

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

        // Derive progress signal from updated_at directly (no heuristics)
        const timeSinceProgress = status.updated_at ? Math.floor((Date.now() - new Date(status.updated_at).getTime()) / 1000) : 0;

        // Progress signal messaging (freshness only)
        let progressMsg = '';
        let progressClass = 'text-gray-500';

        if (status.status === 'RUNNING') {
            if (hasStepFailures) {
                progressMsg = 'A step failed. Waiting for run status to finalize.';
                progressClass = 'text-red-600';
            } else if (timeSinceProgress < 10) {
                progressMsg = timeSinceProgress === 0
                    ? 'Processing is active. System just reported progress.'
                    : `Processing is active. Last progress signal ${timeSinceProgress}s ago.`;
                progressClass = 'text-green-600';
            } else {
                progressMsg = `Last progress signal ${timeSinceProgress}s ago.`;
                progressClass = 'text-gray-600';
            }
        }

        // Find current step details for description
        const currentStepInfo = currentStepVM;
        const currentStepDesc = currentNormalized ? STEP_DESCRIPTIONS[currentNormalized] : null;

        return (
            <div className="p-8 max-w-3xl mx-auto mt-12">
                {/* Header */}
                <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xl font-bold">
                        {vm.overallStatus === 'QUEUED' ? 'Waiting in Queue' :
                            vm.isStale ? 'Run Stalled' : 'Processing Run'}
                    </h2>
                    {vm.isStale ? (
                        <div className="px-3 py-1 bg-orange-100 text-orange-700 rounded text-sm font-semibold border border-orange-300">
                            STALLED (No Heartbeat)
                        </div>
                    ) : (
                        <div className="flex flex-col items-end">
                            <Loader2 className="animate-spin text-blue-600" size={28} />
                            {vm.secondaryReason && (
                                <div className="mt-2 text-xs font-medium text-orange-700 bg-orange-50 px-2 py-1 rounded border border-orange-200">
                                    ⚠️ {vm.secondaryReason}
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Input Context Block */}
                <div className="bg-gray-50 border rounded-lg p-4 mb-6">
                    <h3 className="text-xs font-semibold text-gray-600 mb-2">Input</h3>
                    <div className="space-y-1">
                        <div className="flex items-center gap-2 text-sm">
                            <span>📄</span>
                            <span className="font-medium">
                                {(() => {
                                    const inputName =
                                        (isMeaningful(status.input_metadata?.filename) && status.input_metadata.filename) ||
                                        (isMeaningful(status.input_filename) && status.input_filename) ||
                                        runId;
                                    return inputName;
                                })()}
                            </span>
                        </div>
                        {status.input_metadata?.size_bytes ? (
                            <div className="flex items-center gap-2 text-xs text-gray-500">
                                <span>💾</span>
                                <span>{(status.input_metadata.size_bytes / (1024 * 1024)).toFixed(2)} MB</span>
                            </div>
                        ) : null}
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

                {/* Configuration Block - Uses resolved_config from steps when available */}
                {(() => {
                    const asrStep = status.steps?.find((s: any) => s.name === 'asr');
                    const resolved = asrStep?.resolved_config;

                    return (
                        <div className="bg-gray-50 border rounded-lg p-4 mb-6">
                            <h3 className="text-xs font-semibold text-gray-600 mb-2">Execution Configuration</h3>
                            <div className="grid grid-cols-2 gap-4 text-sm">
                                <div>
                                    <span className="text-gray-500 text-xs block">ASR Model</span>
                                    <span className="font-medium text-gray-800">
                                        {resolved?.model_id ||
                                            (status.config?.asr?.model_name && status.config.asr.model_name !== 'default'
                                                ? status.config.asr.model_name
                                                : <span className="text-orange-600 text-xs">Pending</span>)}
                                    </span>
                                </div>
                                <div>
                                    <span className="text-gray-500 text-xs block">Device</span>
                                    <span className="font-medium text-gray-800">
                                        {(vm.resolvedDevice || resolved?.device)?.toUpperCase() ||
                                            <span className="text-orange-600 text-xs">Pending</span>}
                                    </span>
                                </div>
                                <div>
                                    <span className="text-gray-500 text-xs block">Source</span>
                                    <span className="font-medium text-gray-800">
                                        {resolved?.source ||
                                            <span className="text-orange-600 text-xs">Pending</span>}
                                    </span>
                                </div>
                                <div>
                                    <span className="text-gray-500 text-xs block">Language</span>
                                    <span className="font-medium text-gray-800">
                                        {resolved?.language ||
                                            <span className="text-orange-600 text-xs">Pending</span>}
                                    </span>
                                </div>
                                <div>
                                    <span className="text-gray-500 text-xs block">Diarization</span>
                                    <span className="font-medium text-gray-800">
                                        {status.config?.diarization?.enabled !== false ? 'Enabled' : 'Disabled'}
                                    </span>
                                </div>
                                {status.config?.preprocessing && (
                                    <div className="col-span-2 border-t pt-2 mt-2">
                                        <span className="text-gray-500 text-xs block">Preprocessing</span>
                                        <div className="text-xs text-gray-600 font-mono mt-1">
                                            {status.config.preprocessing.normalize && 'Normalize '}
                                            {status.config.preprocessing.trim_silence && 'Trim '}
                                            {!status.config.preprocessing.normalize && !status.config.preprocessing.trim_silence && 'None'}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    );
                })()}


                {/* Pipeline Progress - Enhanced real-time tracking */}
                {status.steps_progress && status.steps_progress.length > 0 ? (
                    <PipelineProgress 
                        steps={convertStepsProgress(status.steps_progress)} 
                        className="mb-6"
                    />
                ) : (
                    /* Fallback to VM-based visualization if steps_progress not available */
                    <div className="bg-white border rounded-lg p-6 mb-6">
                        <h3 className="text-sm font-semibold text-gray-600 mb-3">Pipeline</h3>
                        <div className="space-y-2">
                            {vm.pipelineSteps.map((step) => {
                                let icon = '○';
                                let textClass = 'text-gray-400';

                                if (step.status === 'COMPLETED') {
                                    icon = '✓';
                                    textClass = 'text-green-600';
                                } else if (step.status === 'FAILED') {
                                    icon = '❌';
                                    textClass = 'text-red-600 font-semibold';
                                } else if (step.status === 'RUNNING') {
                                    icon = '→';
                                    textClass = 'text-blue-600 font-semibold';
                                } else if (step.status === 'NOT_RUN') {
                                    icon = '○';
                                    textClass = 'text-gray-300';
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
                )}

                {/* Artifacts (Semantic Schema) */}
                {renderArtifacts()}


                {/* Step-Level Errors (if any step failed while run is still RUNNING) */}
                {renderStepFailures()}

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
                    <div className="text-xs text-gray-400 font-mono pt-2 border-t space-y-1">
                        <div>Run ID: {runId}</div>
                        {status.updated_at && (
                            <div>Last update: {new Date(status.updated_at).toLocaleTimeString()}</div>
                        )}
                        {status.last_semantic_progress_at && (
                            <div className="text-blue-600 font-semibold">
                                Semantic Progress: {new Date(status.last_semantic_progress_at).toLocaleTimeString()}
                            </div>
                        )}
                    </div>
                </div>

                {/* Agency */}
                <div className="mt-8 text-center flex flex-col items-center gap-4">
                    {/* Actions including Stop */}
                    {renderActions("flex items-center gap-4 justify-center")}

                    <button onClick={onBack} className="mt-2 px-4 py-2 border rounded hover:bg-gray-50 text-sm text-gray-600 bg-white">
                        Back to List
                    </button>
                </div>

                {/* Dev-only debug overlay */}
                {typeof window !== 'undefined' && import.meta.env.DEV && (
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
                                    // secondsSinceProgress, // Removed
                                    // isStalledState, // Removed
                                },
                            }, null, 2)}
                        </pre>
                    </details>
                )}
            </div>
        );
    }

    // 4. Failed (check for partial results but still show failure UI)
    const isFailed = status.status === 'FAILED' || status.status === 'STALE' || status.status === 'CANCELLED';

    if (isFailed && vm) {
        // VM has already determined the failure state
        // Note: failedStepVM computed but may be used for future enhancements
        void vm.pipelineSteps.find(s => s.status === 'FAILED');
        const actionError = getActionableError(status.failed_step, status.error_message);

        return (
            <div className="p-8 max-w-3xl mx-auto mt-12">
                {/* Mode B Error Surface */}
                <div className="bg-red-900/10 border-l-4 border-red-500 p-4 mb-6 rounded-r shadow-sm">
                    <div className="flex justify-between items-start">
                        <div>
                            <h3 className="font-bold text-red-600">
                                {actionError?.title || "Run Failed"}
                            </h3>
                            <p className="text-red-800 mt-1 font-mono text-sm leading-relaxed">
                                {status.error_message || "Unknown error occurred"}
                            </p>
                            {actionError && (
                                <div className="mt-3 bg-red-50 px-3 py-2 rounded text-sm text-red-700 flex items-center gap-2 border border-red-100">
                                    <span>👉</span>
                                    <span className="font-semibold">{actionError.action}</span>
                                </div>
                            )}
                        </div>
                        <div className="text-right space-y-1">
                            <span className="text-xs font-mono text-red-500 bg-red-100 px-2 py-1 rounded border border-red-200 block">
                                Step: {status.failed_step || "N/A"}
                            </span>
                            {status.error_code && (
                                <span className="text-xs font-mono text-red-600 bg-red-50 px-2 py-1 rounded border border-red-200 block">
                                    {status.error_code}
                                </span>
                            )}
                            {status.error_recoverable && (
                                <span className="text-xs bg-yellow-100 text-yellow-800 px-2 py-1 rounded block">
                                    Recoverable - Try Again
                                </span>
                            )}
                        </div>
                    </div>
                </div>

                {/* Input Context Block */}
                <div className="bg-gray-50 border rounded-lg p-4 mb-6">
                    <h3 className="text-xs font-semibold text-gray-600 mb-2">Input</h3>
                    <div className="space-y-1">
                        <div className="flex items-center gap-2 text-sm">
                            <span>📄</span>
                            <span className="font-medium">
                                {(() => {
                                    const inputName =
                                        (isMeaningful(status.input_metadata?.filename) && status.input_metadata.filename) ||
                                        (isMeaningful(status.input_filename) && status.input_filename) ||
                                        runId;
                                    return inputName;
                                })()}
                            </span>
                        </div>
                        {status.input_metadata?.size_bytes ? (
                            <div className="flex items-center gap-2 text-xs text-gray-500">
                                <span>💾</span>
                                <span>{(status.input_metadata.size_bytes / (1024 * 1024)).toFixed(2)} MB</span>
                            </div>
                        ) : null}
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

                {/* Artifacts (persist even in FAILED view when steps completed) */}
                {renderArtifacts()}
                {/* Step Failures (persist even in FAILED view) */}
                {renderStepFailures()}

                {/* Error Message Coarse (always show for failed runs) */}
                {vm && (vm.primaryReason || vm.secondaryReason) && (
                    <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
                        {status.error_code && (
                            <div className="text-sm font-semibold text-red-900 mb-1">
                                {status.error_code}
                            </div>
                        )}
                        <div className="text-sm text-red-700">
                            {vm.primaryReason}
                        </div>
                        {vm.secondaryReason && (
                            <div className="text-sm text-red-600 mt-1 italic">
                                {vm.secondaryReason}
                            </div>
                        )}
                    </div>
                )}



                {/* Pipeline visualization with failed step */}
                {vm && (
                <div className="bg-white border rounded-lg p-6 mb-4">
                    <h3 className="text-sm font-semibold text-gray-600 mb-3">Pipeline</h3>
                    <div className="space-y-2">
                        <div className="space-y-2">
                            {vm.pipelineSteps.map((step) => {
                                let icon = '○';
                                let textClass = 'text-gray-400';

                                if (step.status === 'COMPLETED') {
                                    icon = '✓';
                                    textClass = 'text-green-600';
                                } else if (step.status === 'FAILED') {
                                    icon = '❌';
                                    textClass = 'text-red-600 font-semibold';
                                } else if (step.status === 'RUNNING') {
                                    icon = '→';
                                    textClass = 'text-blue-600 font-semibold';
                                } else if (step.status === 'NOT_RUN') {
                                    icon = '○';
                                    textClass = 'text-gray-300';
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
                </div>
                )}

                {/* Explicit unknown state */}
                {/* 
                  VM handles all states. No need for inference disclaimer.
                  If UNKNOWN, it shows as gray circle.
                */}

                {/* Metadata */}
                <div className="bg-gray-50 rounded-lg p-4 mb-6">
                    <div className="text-xs text-gray-400 font-mono">
                        Run ID: {runId}
                    </div>
                </div>

                {/* Actions */}
                {/* Actions */}
                <div className="mt-8 flex flex-col items-center gap-4">
                    {renderActions("flex items-center gap-4 justify-center")}
                    <button type="button" onClick={onBack} className="mt-2 text-sm text-gray-500 hover:text-gray-700 underline">
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
                    <button type="button" onClick={onBack} className="p-2 hover:bg-gray-100 rounded-full" title="Back to list" aria-label="Back to list">
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
                {renderActions("flex gap-2")}
            </header>

            {/* Content Scroller */}
            <div className="flex-1 overflow-y-auto p-8">
                <div className="max-w-3xl mx-auto bg-white rounded-lg shadow-sm border p-8 min-h-[50vh]">
                    {!detail ? (
                        <div className="flex items-center gap-2 text-gray-500">
                            <Loader2 className="animate-spin" size={20} />
                            Loading transcript details...
                        </div>
                    ) : (
                        <div className="space-y-4">
                            {/* Transcript View - Rendered here */}
                            {(detail.segments || []).length === 0 ? (
                                <div className="text-gray-400 text-center italic">No transcript segments found.</div>
                            ) : (
                                detail.segments.map((seg, idx) => (
                                    <div key={idx} className="flex gap-4">
                                        <div className="w-16 flex-shrink-0 text-xs text-gray-400 font-mono mt-1">
                                            {seg.start_s != null ? `${Math.floor(seg.start_s / 60)}:${String(Math.floor(seg.start_s % 60)).padStart(2, '0')}` : '--:--'}
                                        </div>
                                        <div className="flex-1">
                                            {seg.speaker && <div className="font-bold text-xs text-gray-600 mb-0.5">{seg.speaker}</div>}
                                            <p className="text-gray-800 leading-relaxed">{seg.text}</p>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    )}
                </div>

                {/* Run History Panel */}
                {status?.input_hash && (
                    <div className="max-w-3xl mx-auto mt-6">
                        <RunHistory
                            currentRunId={runId}
                            inputHash={status.input_hash}
                            onSelectRun={(id) => window.location.href = `/runs/${id}`}
                        />
                    </div>
                )}

                <DebugPanel run={status} />
            </div>
        </div>
    );
}
