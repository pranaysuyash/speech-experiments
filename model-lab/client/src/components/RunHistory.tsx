import { useState, useEffect } from 'react';
import { api } from '../lib/api';
import type { RunSummary } from '../lib/api';
import { History, GitCompare, RefreshCw } from 'lucide-react';

interface RunHistoryProps {
    currentRunId: string;
    inputHash?: string;
    onSelectRun?: (runId: string) => void;
}

export default function RunHistory({ currentRunId, inputHash, onSelectRun }: RunHistoryProps) {
    const [relatedRuns, setRelatedRuns] = useState<RunSummary[]>([]);
    const [loading, setLoading] = useState(false);
    const [selectedForCompare, setSelectedForCompare] = useState<Set<string>>(new Set());
    const [expanded, setExpanded] = useState(false);

    useEffect(() => {
        if (inputHash && expanded) {
            loadRelatedRuns();
        }
    }, [inputHash, expanded]);

    const loadRelatedRuns = async () => {
        if (!inputHash) return;
        setLoading(true);
        try {
            const runs = await api.getRunsByInput(inputHash);
            setRelatedRuns(runs);
        } catch (error) {
            console.error('Failed to load related runs:', error);
        } finally {
            setLoading(false);
        }
    };

    const toggleCompareSelection = (runId: string) => {
        const newSelected = new Set(selectedForCompare);
        if (newSelected.has(runId)) {
            newSelected.delete(runId);
        } else {
            if (newSelected.size < 2) {
                newSelected.add(runId);
            }
        }
        setSelectedForCompare(newSelected);
    };

    const handleCompare = () => {
        const ids = Array.from(selectedForCompare);
        if (ids.length === 2) {
            window.location.href = `/compare?a=${ids[0]}&b=${ids[1]}`;
        }
    };

    const handleRerun = async () => {
        try {
            const result = await api.rerunPipeline(currentRunId);
            window.location.href = result.console_url;
        } catch (error) {
            console.error('Failed to rerun:', error);
            alert('Failed to start rerun. The runner may be busy.');
        }
    };

    if (!inputHash) {
        return null;
    }

    const statusColors: Record<string, string> = {
        COMPLETED: 'bg-green-100 text-green-800',
        FAILED: 'bg-red-100 text-red-800',
        RUNNING: 'bg-blue-100 text-blue-800',
        STALE: 'bg-orange-100 text-orange-700',
        PENDING: 'bg-gray-100 text-gray-600',
    };

    return (
        <div className="border rounded-lg bg-white">
            <button
                onClick={() => setExpanded(!expanded)}
                className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50"
            >
                <div className="flex items-center gap-2">
                    <History size={18} className="text-gray-500" />
                    <span className="font-medium">Run History</span>
                </div>
                <span className="text-gray-400">{expanded ? '▼' : '▶'}</span>
            </button>

            {expanded && (
                <div className="border-t px-4 py-3">
                    {loading ? (
                        <div className="text-sm text-gray-500">Loading related runs...</div>
                    ) : relatedRuns.length === 0 ? (
                        <div className="text-sm text-gray-500">No other runs found for this input.</div>
                    ) : (
                        <>
                            <div className="text-xs text-gray-500 mb-2">
                                {relatedRuns.length} run{relatedRuns.length !== 1 ? 's' : ''} with same input
                            </div>

                            <div className="space-y-2 max-h-64 overflow-y-auto">
                                {relatedRuns.map((run) => {
                                    const isCurrent = run.run_id === currentRunId;
                                    const isSelected = selectedForCompare.has(run.run_id);

                                    return (
                                        <div
                                            key={run.run_id}
                                            className={`flex items-center gap-2 p-2 rounded border ${isCurrent ? 'border-blue-300 bg-blue-50' : 'border-gray-200'
                                                }`}
                                        >
                                            <input
                                                type="checkbox"
                                                checked={isSelected}
                                                onChange={() => toggleCompareSelection(run.run_id)}
                                                disabled={!isSelected && selectedForCompare.size >= 2}
                                                className="w-4 h-4"
                                            />
                                            <div
                                                className="flex-1 cursor-pointer"
                                                onClick={() => onSelectRun?.(run.run_id)}
                                            >
                                                <div className="flex items-center gap-2">
                                                    <span
                                                        className={`px-1.5 py-0.5 rounded text-xs font-medium ${statusColors[run.status] || 'bg-gray-100 text-gray-800'
                                                            }`}
                                                    >
                                                        {run.status}
                                                    </span>
                                                    {isCurrent && (
                                                        <span className="text-xs text-blue-600 font-medium">
                                                            (current)
                                                        </span>
                                                    )}
                                                </div>
                                                <div className="text-xs text-gray-500 mt-1">
                                                    {run.started_at
                                                        ? new Date(run.started_at).toLocaleString()
                                                        : 'Unknown date'}
                                                </div>
                                                <div className="text-xs text-gray-400 font-mono truncate">
                                                    {run.run_id.slice(0, 16)}...
                                                </div>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>

                            <div className="flex gap-2 mt-3 pt-3 border-t">
                                <button
                                    onClick={handleCompare}
                                    disabled={selectedForCompare.size !== 2}
                                    className={`flex items-center gap-1 px-3 py-1.5 text-sm rounded ${selectedForCompare.size === 2
                                            ? 'bg-purple-600 text-white hover:bg-purple-700'
                                            : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                                        }`}
                                >
                                    <GitCompare size={14} />
                                    Compare Selected
                                </button>
                                <button
                                    onClick={handleRerun}
                                    className="flex items-center gap-1 px-3 py-1.5 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
                                >
                                    <RefreshCw size={14} />
                                    Rerun
                                </button>
                            </div>
                        </>
                    )}
                </div>
            )}
        </div>
    );
}
