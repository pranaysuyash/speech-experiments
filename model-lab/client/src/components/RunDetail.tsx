import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { api } from '../lib/api';
import type { RunDetail as RunDetailType, ResultSummary } from '../lib/api';
import { Loader2, ArrowLeft, Trash2, Download } from 'lucide-react';

interface RunDetailProps {
    onBack?: () => void;
}

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

    // 1. Missing ID
    if (!runId) return <div className="p-8">Missing run id</div>;

    // 2. Initial Loading
    if (!status) return <div className="p-8 flex items-center gap-2"><Loader2 className="animate-spin" /> Loading run status...</div>;

    // 3. Queued / Running
    if (status.status === 'QUEUED' || status.status === 'RUNNING') {
        // Simple human mapping for steps
        const stepMap: Record<string, string> = {
            'ingest': 'Ingesting Media',
            'vad': 'Detecting Voice Activity',
            'diarization': 'Identifying Speakers',
            'transcription': 'Transcribing Audio',
            'metrics': 'Computing Metrics'
        };
        const readableStep = status.current_step
            ? (stepMap[status.current_step.toLowerCase()] || status.current_step)
            : 'Initializing...';

        // Static expectation framing per step
        const expectationMap: Record<string, string> = {
            'ingest': 'Typically completes in seconds',
            'vad': 'Usually takes under 30 seconds',
            'diarization': 'May take 1-2 minutes for longer audio',
            'transcription': 'Typical transcription takes under a minute for short audio',
            'metrics': 'Final step, usually completes quickly'
        };
        const expectation = status.current_step
            ? (expectationMap[status.current_step.toLowerCase()] || 'Processing...')
            : 'Runs typically complete in a few minutes';

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

        // Heartbeat freshness
        let freshness = '';
        if (lastPolledAt > 0) {
            const sinceLastPoll = Math.floor((Date.now() - lastPolledAt) / 1000);
            if (sinceLastPoll < 60) {
                freshness = `${sinceLastPoll}s ago`;
            } else {
                freshness = `${Math.floor(sinceLastPoll / 60)}m ago`;
            }
        }

        return (
            <div className="p-8 max-w-2xl mx-auto text-center mt-20">
                <Loader2 className="animate-spin mx-auto mb-6 text-blue-600" size={48} />

                <h2 className="text-xl font-bold mb-2">
                    {status.status === 'QUEUED' ? 'Waiting in Queue' : 'Processing Run'}
                </h2>

                <div className="text-lg text-gray-700 font-medium mb-1">
                    {readableStep}
                </div>

                {elapsed && (
                    <p className="text-sm text-gray-500 font-mono mb-1">
                        Time Elapsed: {elapsed}
                    </p>
                )}

                {freshness && (
                    <p className="text-xs text-gray-400 font-mono">
                        Last update: {freshness}
                    </p>
                )}

                <div className="mt-4 text-sm text-gray-600 italic">
                    {expectation}
                </div>

                <div className="mt-12 text-xs text-gray-400 font-mono bg-gray-50 inline-block px-3 py-1 rounded">
                    ID: {runId}
                </div>

                <div className="mt-6">
                    <button onClick={onBack} className="px-4 py-2 border rounded hover:bg-gray-50 text-sm text-gray-600">
                        Back to List
                    </button>
                </div>
            </div>
        );
    }

    // 4. Failed (and not partial)
    const isFailed = status.status === 'FAILED' || status.status === 'STALE';
    const showPartial = isFailed && result?.quality_flags.is_partial;

    if (isFailed && !showPartial) {
        return (
            <div className="p-8 max-w-2xl mx-auto text-center mt-20">
                <div className="mx-auto mb-4 w-12 h-12 bg-red-100 text-red-600 rounded-full flex items-center justify-center">
                    <Trash2 size={24} />
                </div>
                <h2 className="text-xl font-bold mb-2 text-red-700">Run Failed</h2>
                <p className="text-gray-800 font-medium">{status.error_code}</p>
                <p className="text-gray-600 mt-2">{status.error_message}</p>
                <div className="mt-8 text-sm text-gray-400 font-mono">Run ID: {runId}</div>
                <button onClick={onBack} className="mt-8 px-4 py-2 border rounded hover:bg-gray-50">Back to List</button>
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
                                    {result.metrics.duration_s !== undefined && (
                                        <span>⏱ {result.metrics.duration_s.toFixed(1)}s</span>
                                    )}
                                    {result.metrics.word_count !== undefined && (
                                        <span>📝 {result.metrics.word_count} words</span>
                                    )}
                                    {result.metrics.confidence_avg !== undefined && (
                                        <span>🎯 {(result.metrics.confidence_avg * 100).toFixed(1)}%</span>
                                    )}
                                </>
                            )}
                        </div>
                    </div>
                </div>

                {/* Actions */}
                <div className="flex gap-2">
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
                                        {formatTime(seg.start_s)}
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
