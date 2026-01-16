import { useEffect, useState } from 'react';
import { api } from '../lib/api';
import type { RunSummary } from '../lib/api';
import { Clock, CheckCircle } from 'lucide-react';

interface RunsListProps {
    onSelectRun: (runId: string) => void;
}

export function RunsList({ onSelectRun }: RunsListProps) {
    const [runs, setRuns] = useState<RunSummary[]>([]);
    const [loading, setLoading] = useState(true);

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
            setRuns(data);
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

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {runs.map(run => (
                    <div
                        key={run.run_id}
                        className="border rounded-lg p-4 hover:shadow-lg transition-shadow cursor-pointer bg-white"
                        onClick={() => onSelectRun(run.run_id)}
                    >
                        <div className="flex justify-between items-start mb-2">
                            <h3 className="font-semibold text-lg truncate max-w-[200px]" title={run.input_filename}>
                                {run.input_filename}
                            </h3>
                            <StatusBadge status={run.status} />
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
                ))}
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
            )}
        </div>
    );
}

function StatusBadge({ status }: { status: string }) {
    const colors = {
        COMPLETED: 'bg-green-100 text-green-800',
        FAILED: 'bg-red-100 text-red-800',
        RUNNING: 'bg-blue-100 text-blue-800',
        PENDING: 'bg-gray-100 text-gray-600'
    };
    const color = colors[status as keyof typeof colors] || 'bg-gray-100 text-gray-800';

    return (
        <span className={`px-2 py-0.5 rounded text-xs font-bold ${color}`}>
            {status}
        </span>
    )
}
