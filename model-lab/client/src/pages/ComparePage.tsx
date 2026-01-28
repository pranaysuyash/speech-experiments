import { useState, useEffect } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import { api } from '../lib/api';
import type { RunComparison } from '../lib/api';
import { ArrowUpDown } from 'lucide-react';

export default function ComparePage() {
    const [searchParams] = useSearchParams();
    const runA = searchParams.get('a');
    const runB = searchParams.get('b');

    const [comparison, setComparison] = useState<RunComparison | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (runA && runB) {
            loadComparison();
        } else {
            setError('Both run IDs (a and b) are required');
            setLoading(false);
        }
    }, [runA, runB]);

    const loadComparison = async () => {
        if (!runA || !runB) return;
        setLoading(true);
        setError(null);
        try {
            const data = await api.compareRuns(runA, runB);
            setComparison(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load comparison');
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="p-8 text-center text-gray-500">
                Loading comparison...
            </div>
        );
    }

    if (error) {
        return (
            <div className="p-8">
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
                    {error}
                </div>
                <Link to="/lab/runs" className="mt-4 inline-block text-blue-600 hover:underline">
                    ← Back to Runs
                </Link>
            </div>
        );
    }

    if (!comparison) {
        return null;
    }

    const statusColors: Record<string, string> = {
        COMPLETED: 'bg-green-100 text-green-800',
        FAILED: 'bg-red-100 text-red-800',
        RUNNING: 'bg-blue-100 text-blue-800',
        STALE: 'bg-orange-100 text-orange-700',
        PENDING: 'bg-gray-100 text-gray-600',
    };

    const getDiffBadge = (diff: number | null) => {
        if (diff === null || diff === 0) {
            return <span className="text-gray-400">—</span>;
        }
        const isPositive = diff > 0;
        return (
            <span className={`font-medium ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
                {isPositive ? '+' : ''}{typeof diff === 'number' ? diff.toLocaleString() : diff}
            </span>
        );
    };

    const formatValue = (val: number | null): string => {
        if (val === null || val === undefined) return '—';
        if (typeof val === 'number') {
            return val.toLocaleString();
        }
        return String(val);
    };

    return (
        <div className="p-6 max-w-6xl mx-auto">
            <div className="mb-6">
                <Link to="/lab/runs" className="text-blue-600 hover:underline text-sm">
                    ← Back to Runs
                </Link>
                <h1 className="text-2xl font-bold mt-2">Run Comparison</h1>
            </div>

            {/* Run Summaries - Split View */}
            <div className="grid grid-cols-2 gap-4 mb-6">
                {/* Run A */}
                <div className="border rounded-lg p-4 bg-white">
                    <div className="flex items-center justify-between mb-3">
                        <span className="text-lg font-semibold text-purple-600">Run A</span>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${statusColors[comparison.runs.a.status] || 'bg-gray-100'}`}>
                            {comparison.runs.a.status}
                        </span>
                    </div>
                    <div className="space-y-2 text-sm">
                        <div>
                            <span className="text-gray-500">Run ID:</span>
                            <Link
                                to={`/runs/${comparison.runs.a.run_id}`}
                                className="ml-2 font-mono text-blue-600 hover:underline"
                            >
                                {comparison.runs.a.run_id.slice(0, 20)}...
                            </Link>
                        </div>
                        <div>
                            <span className="text-gray-500">File:</span>
                            <span className="ml-2">{comparison.runs.a.input_filename || '—'}</span>
                        </div>
                        <div>
                            <span className="text-gray-500">Started:</span>
                            <span className="ml-2">
                                {comparison.runs.a.started_at
                                    ? new Date(comparison.runs.a.started_at).toLocaleString()
                                    : '—'}
                            </span>
                        </div>
                    </div>
                </div>

                {/* Run B */}
                <div className="border rounded-lg p-4 bg-white">
                    <div className="flex items-center justify-between mb-3">
                        <span className="text-lg font-semibold text-indigo-600">Run B</span>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${statusColors[comparison.runs.b.status] || 'bg-gray-100'}`}>
                            {comparison.runs.b.status}
                        </span>
                    </div>
                    <div className="space-y-2 text-sm">
                        <div>
                            <span className="text-gray-500">Run ID:</span>
                            <Link
                                to={`/runs/${comparison.runs.b.run_id}`}
                                className="ml-2 font-mono text-blue-600 hover:underline"
                            >
                                {comparison.runs.b.run_id.slice(0, 20)}...
                            </Link>
                        </div>
                        <div>
                            <span className="text-gray-500">File:</span>
                            <span className="ml-2">{comparison.runs.b.input_filename || '—'}</span>
                        </div>
                        <div>
                            <span className="text-gray-500">Started:</span>
                            <span className="ml-2">
                                {comparison.runs.b.started_at
                                    ? new Date(comparison.runs.b.started_at).toLocaleString()
                                    : '—'}
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Config Diff */}
            <div className="border rounded-lg bg-white mb-6">
                <div className="px-4 py-3 border-b bg-gray-50 font-medium">
                    Configuration Differences
                </div>
                <div className="p-4">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b">
                                <th className="text-left py-2 text-gray-500">Setting</th>
                                <th className="text-left py-2 text-purple-600">Run A</th>
                                <th className="text-left py-2 text-indigo-600">Run B</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr className="border-b">
                                <td className="py-2 text-gray-700">Steps</td>
                                <td className="py-2">
                                    <div className="flex flex-wrap gap-1">
                                        {comparison.config_diff.steps.a.map((s) => (
                                            <span key={s} className="px-1.5 py-0.5 bg-purple-100 text-purple-700 rounded text-xs">
                                                {s}
                                            </span>
                                        ))}
                                    </div>
                                </td>
                                <td className="py-2">
                                    <div className="flex flex-wrap gap-1">
                                        {comparison.config_diff.steps.b.map((s) => (
                                            <span key={s} className="px-1.5 py-0.5 bg-indigo-100 text-indigo-700 rounded text-xs">
                                                {s}
                                            </span>
                                        ))}
                                    </div>
                                </td>
                            </tr>
                            <tr>
                                <td className="py-2 text-gray-700">Preprocessing</td>
                                <td className="py-2">
                                    {comparison.config_diff.preprocessing.a.length > 0 ? (
                                        <div className="flex flex-wrap gap-1">
                                            {comparison.config_diff.preprocessing.a.map((p) => (
                                                <span key={p} className="px-1.5 py-0.5 bg-gray-100 text-gray-700 rounded text-xs">
                                                    {p}
                                                </span>
                                            ))}
                                        </div>
                                    ) : (
                                        <span className="text-gray-400">None</span>
                                    )}
                                </td>
                                <td className="py-2">
                                    {comparison.config_diff.preprocessing.b.length > 0 ? (
                                        <div className="flex flex-wrap gap-1">
                                            {comparison.config_diff.preprocessing.b.map((p) => (
                                                <span key={p} className="px-1.5 py-0.5 bg-gray-100 text-gray-700 rounded text-xs">
                                                    {p}
                                                </span>
                                            ))}
                                        </div>
                                    ) : (
                                        <span className="text-gray-400">None</span>
                                    )}
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Metrics Comparison */}
            <div className="border rounded-lg bg-white">
                <div className="px-4 py-3 border-b bg-gray-50 font-medium flex items-center gap-2">
                    <ArrowUpDown size={16} />
                    Metrics Comparison
                </div>
                <div className="p-4">
                    {Object.keys(comparison.metrics_comparison).length === 0 ? (
                        <div className="text-gray-500 text-center py-4">
                            No metrics available for comparison
                        </div>
                    ) : (
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b">
                                    <th className="text-left py-2 text-gray-500">Metric</th>
                                    <th className="text-right py-2 text-purple-600">Run A</th>
                                    <th className="text-right py-2 text-indigo-600">Run B</th>
                                    <th className="text-right py-2 text-gray-500">Δ (B - A)</th>
                                </tr>
                            </thead>
                            <tbody>
                                {Object.entries(comparison.metrics_comparison).map(([key, values]) => (
                                    <tr key={key} className="border-b last:border-b-0">
                                        <td className="py-3 text-gray-700">
                                            {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                        </td>
                                        <td className="py-3 text-right font-mono">
                                            {formatValue(values.a)}
                                        </td>
                                        <td className="py-3 text-right font-mono">
                                            {formatValue(values.b)}
                                        </td>
                                        <td className="py-3 text-right">
                                            {getDiffBadge(values.diff)}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    )}
                </div>
            </div>
        </div>
    );
}
