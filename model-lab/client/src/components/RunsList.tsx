import { useEffect, useState } from 'react';
import { api } from '../lib/api';
import type { RunSummary } from '../lib/api';
import { Clock, CheckCircle } from 'lucide-react';
import { sortRuns } from '../lib/runSorting';
import { deriveProgressSignal, isStalled } from '../lib/runProgress';

interface RunsListProps {
    onSelectRun: (runId: string) => void;
}

export default function RunsList({ onSelectRun }: RunsListProps) {
    const [runs, setRuns] = useState<RunSummary[]>([]);
    const [loading, setLoading] = useState(true);
    const [lastActiveRunId, setLastActiveRunId] = useState<string | null>(
        localStorage.getItem('lastActiveRunId')
    );

    useEffect(() => {
        loadRuns();
    }, []);

    const handleRefresh = async () => {
        setLoading(true);
        try {
            // Force server-side scan
            await api.refreshRuns();
            // Then reload list
            const data = await api.getRuns();
            setRuns(data);
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    const loadRuns = async () => {
        setLoading(true);
        try {
            const data = await api.getRuns();
            // Use centralized sorting (depends only on API data, never UI state)
            const sorted = sortRuns(data);
            setRuns(sorted);
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    // Status polling for RUNNING runs
    // Uses single timer for efficiency, auto-stops when no RUNNING runs remain
    useEffect(() => {
        const hasRunning = runs.some((r) => r.status === 'RUNNING');
        if (!hasRunning) return;

        let inFlight = false;

        const intervalId = window.setInterval(async () => {
            if (inFlight) return;
            inFlight = true;

            try {
                const runningRuns = runs.filter((r) => r.status === 'RUNNING');
                if (runningRuns.length === 0) return;

                // Batch-fetch status for all RUNNING runs
                const updates = await Promise.all(
                    runningRuns.map(async (r) => {
                        try {
                            const s = await api.getRunStatus(r.run_id);
                            return { run_id: r.run_id, status: s.status, current_step: s.current_step };
                        } catch {
                            return null;
                        }
                    })
                );

                const nonNullUpdates = updates.filter(
                    (u): u is { run_id: string; status: string; current_step: string | null | undefined } => u !== null
                );
                const byId = new Map(
                    nonNullUpdates.map((u) => [u.run_id, { status: u.status, current_step: u.current_step }])
                );

                // Update state only if status changed
                setRuns((prev) =>
                    prev.map((r) => {
                        const update = byId.get(r.run_id);
                        if (!update || update.status === r.status) return r;
                        return { ...r, status: update.status };
                    })
                );
            } finally {
                inFlight = false;
            }
        }, 1500); // Poll every 1.5s

        return () => {
            window.clearInterval(intervalId);
        };
    }, [runs]);

    if (loading) return <div className="p-8 text-center text-gray-500">Loading runs...</div>;

    return (
        <div className="p-6">
            <div className="flex justify-between items-center mb-6">
                <h1 className="text-2xl font-bold">Analysis Runs</h1>
                <button onClick={handleRefresh} className="text-sm text-blue-500 hover:underline">Refresh Index</button>
            </div>

            {/* Last Run Summary Panel - Shows most recently STARTED run */}
            {runs.length > 0 && (() => {
                // Find most recently started run (max started_at)
                const lastStartedRun = runs.reduce((latest, run) => {
                    if (!run.started_at) return latest;
                    if (!latest || !latest.started_at) return run;
                    return new Date(run.started_at) > new Date(latest.started_at) ? run : latest;
                }, runs[0]);

                if (!lastStartedRun) return null;

                return (
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
                        <div className="flex items-center justify-between">
                            <div>
                                <h3 className="text-sm font-semibold text-blue-900 mb-1">Last Started Run</h3>
                                <div className="space-y-1">
                                    <div className="text-sm text-blue-800">
                                        📄 {lastStartedRun.input_filename}
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <StatusBadge status={lastStartedRun.status} />
                                    </div>
                                </div>
                            </div>
                            <div className="flex gap-2">
                                <button
                                    onClick={() => onSelectRun(lastStartedRun.run_id)}
                                    className="px-3 py-1.5 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
                                >
                                    View
                                </button>
                                <button
                                    onClick={() => {
                                        const params = new URLSearchParams({
                                            repeat_from: lastStartedRun.run_id,
                                            file: lastStartedRun.input_filename || '',
                                        });
                                        window.location.href = `/lab/workbench?${params}`;
                                    }}
                                    className="px-3 py-1.5 text-sm border border-blue-600 text-blue-600 rounded hover:bg-blue-50"
                                >
                                    Repeat
                                </button>
                            </div>
                        </div>
                    </div>
                );
            })()}

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {runs.map(run => {
                    const isLastActive = run.run_id === lastActiveRunId;

                    // Detect STALLED state for display (centralized helper)
                    const { secondsSinceProgress } = deriveProgressSignal(run.updated_at);
                    const effectiveStatus = isStalled(run.status, secondsSinceProgress) ? 'STALLED' : run.status;

                    return (
                        <div
                            key={run.run_id}
                            className={`border rounded-lg p-4 hover:shadow-lg transition-shadow cursor-pointer bg-white ${isLastActive ? 'ring-2 ring-blue-400' : ''
                                }`}
                            onClick={() => {
                                localStorage.setItem('lastActiveRunId', run.run_id);
                                setLastActiveRunId(run.run_id);
                                onSelectRun(run.run_id);
                            }}
                        >
                            <div className="flex justify-between items-start mb-2">
                                <h3 className="font-semibold text-lg truncate max-w-[200px]" title={run.input_filename}>
                                    {run.input_filename}
                                </h3>
                                <div className="flex gap-1">
                                    {isLastActive && (
                                        <span className="px-2 py-0.5 rounded text-xs font-bold bg-blue-100 text-blue-800">
                                            Last active
                                        </span>
                                    )}
                                    <StatusBadge status={effectiveStatus} />
                                </div>
                            </div>

                            <div className="text-xs text-gray-500 mb-4 font-mono">
                                ID: {run.run_id}
                            </div>

                            <div className="flex items-center gap-4 text-sm text-gray-600 mb-3">
                                <div className="flex items-center gap-1">
                                    <Clock size={16} />
                                    <span>{run.duration ? `${run.duration.toFixed(1)}s` : '--'}</span>
                                </div>
                                <div className="flex items-center gap-1">
                                    <CheckCircle size={16} />
                                    <span>{run.steps_completed.length} Steps</span>
                                </div>
                            </div>

                            <div className="flex justify-end pt-2 border-t">
                                <span className="text-blue-600 font-medium text-sm flex items-center gap-1">
                                    View Details →
                                </span>
                            </div>
                        </div>
                    );
                })}
            </div>

            {runs.length === 0 && (
                <div className="text-center py-12 text-gray-400">
                    <p className="mb-4">No runs found locally.</p>
                    <button
                        onClick={handleRefresh}
                        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                    >
                        Scan for Runs
                    </button>
                </div>
            )
            }
        </div >
    );
}

function StatusBadge({ status }: { status: string }) {
    const colors = {
        COMPLETED: 'bg-green-100 text-green-800',
        FAILED: 'bg-red-100 text-red-800',
        RUNNING: 'bg-blue-100 text-blue-800',
        STALLED: 'bg-orange-100 text-orange-700 border border-orange-300',
        PENDING: 'bg-gray-100 text-gray-600'
    };
    const color = colors[status as keyof typeof colors] || 'bg-gray-100 text-gray-800';

    return (
        <span className={`px-2 py-0.5 rounded text-xs font-bold ${color}`}>
            {status}
        </span>
    )
}
