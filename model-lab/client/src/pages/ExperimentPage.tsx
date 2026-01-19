import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { api } from '../lib/api';
import type { ComparisonSummary } from '../lib/api';

interface ExperimentRun {
    candidate_id: string;
    steps_preset: string;
    run_id: string | null;
    status: 'QUEUED' | 'STARTING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'STALE';
    created_at: string;
    started_at: string | null;
    ended_at: string | null;
    score_cards?: { name: string; label: string; score: number; type: string }[];
}

interface Experiment {
    experiment_id: string;
    use_case_id: string;
    created_at: string;
    source: {
        filename_original: string;
        bytes: number;
        sha256: string;
    };
    candidates: { candidate_id: string; label: string; steps_preset: string }[];
    runs: ExperimentRun[];
    last_updated_at: string;
}

const STATUS_COLORS: Record<string, { bg: string; text: string }> = {
    QUEUED: { bg: '#f3f4f6', text: '#6b7280' },
    STARTING: { bg: '#fef3c7', text: '#d97706' },
    RUNNING: { bg: '#dbeafe', text: '#2563eb' },
    COMPLETED: { bg: '#d1fae5', text: '#059669' },
    FAILED: { bg: '#fee2e2', text: '#dc2626' },
    STALE: { bg: '#fef9c3', text: '#ca8a04' },
};

export default function ExperimentPage() {
    const { experimentId } = useParams<{ experimentId: string }>();
    const [experiment, setExperiment] = useState<Experiment | null>(null);
    const [error, setError] = useState<string | null>(null);

    // Comparison state
    const [comparison, setComparison] = useState<ComparisonSummary | null>(null);
    const [compLoading, setCompLoading] = useState(false);
    const [compError, setCompError] = useState<string | null>(null);

    // Sort state
    const [sortConfig, setSortConfig] = useState<{ key: string; direction: 'asc' | 'desc' }>({
        key: 'candidate_id',
        direction: 'asc'
    });

    useEffect(() => {
        if (!experimentId) return;

        const load = async () => {
            try {
                const data = await api.getExperiment(experimentId);
                setExperiment(data);

                // Load comparison if strictly 2 runs or just try it
                if (data.runs.length > 0) {
                    setCompLoading(true);
                    try {
                        const comp = await api.getExperimentComparisonResults(experimentId);
                        setComparison(comp);
                    } catch (err) {
                        console.error("Comparison load failed", err);
                        setCompError("Could not load semantic comparison.");
                    } finally {
                        setCompLoading(false);
                    }
                }
            } catch (e: any) {
                setError(e?.message || 'Failed to load experiment');
            }
        };

        load();
    }, [experimentId]);

    // Polling updates logic
    useEffect(() => {
        if (!experimentId) return;
        const interval = setInterval(async () => {
            const data = await api.getExperiment(experimentId);
            setExperiment(data);
            // Optionally refresh comparison too if status changed?
            // For V1 let's just refresh comparison every time we poll experiment, 
            // or only if we know it's pending.
            // Simplicity: Refresh both.
            try {
                const comp = await api.getExperimentComparisonResults(experimentId);
                setComparison(comp);
            } catch (e) { }
        }, 3000);
        return () => clearInterval(interval);
    }, [experimentId]);

    if (error) {
        return (
            <div style={{ padding: '2rem' }}>
                <div style={{ color: '#dc2626', background: '#fee2e2', padding: '1rem', borderRadius: 8 }}>
                    {error}
                </div>
            </div>
        );
    }

    if (!experiment) {
        return (
            <div style={{ padding: '2rem' }}>
                <div>Loading experiment...</div>
            </div>
        );
    }

    // Compute unique score columns
    const scoreColumns = Array.from(new Set(
        experiment.runs.flatMap(r => r.score_cards?.map(s => s.name) || [])
    )).sort();

    // Map name to label (find first occurrence)
    const scoreLabels: Record<string, string> = {};
    const scoreTypes: Record<string, string> = {};
    experiment.runs.forEach(r => {
        r.score_cards?.forEach(s => {
            if (!scoreLabels[s.name]) scoreLabels[s.name] = s.label;
            if (!scoreTypes[s.name]) scoreTypes[s.name] = s.type;
        });
    });

    // Sorting logic
    const sortedRuns = [...experiment.runs].sort((a, b) => {
        let valA: any = a[sortConfig.key as keyof ExperimentRun];
        let valB: any = b[sortConfig.key as keyof ExperimentRun];

        // Handle score keys
        if (sortConfig.key.startsWith('score:')) {
            const scoreName = sortConfig.key.split(':')[1];
            valA = a.score_cards?.find(s => s.name === scoreName)?.score ?? -1;
            valB = b.score_cards?.find(s => s.name === scoreName)?.score ?? -1;
        }

        if (valA < valB) return sortConfig.direction === 'asc' ? -1 : 1;
        if (valA > valB) return sortConfig.direction === 'asc' ? 1 : -1;
        return 0;
    });

    const handleSort = (key: string) => {
        setSortConfig(current => ({
            key,
            direction: current.key === key && current.direction === 'asc' ? 'desc' : 'asc'
        }));
    };

    return (
        <div style={{ padding: '2rem' }}>
            {/* Header */}
            <div style={{ marginBottom: '1.5rem' }}>
                <h1 style={{ fontSize: '1.4rem', fontWeight: 700, marginBottom: '0.5rem' }}>
                    Experiment: {experiment.experiment_id}
                </h1>
                <div style={{ fontSize: '0.875rem', color: '#6b7280' }}>
                    <span>Use Case: {experiment.use_case_id}</span>
                    <span style={{ margin: '0 0.5rem' }}>•</span>
                    <span>File: {experiment.source.filename_original}</span>
                    <span style={{ margin: '0 0.5rem' }}>•</span>
                    <span>Created: {new Date(experiment.created_at).toLocaleString()}</span>
                </div>
            </div>

            {/* Candidates Table */}
            <div style={{ marginBottom: '2rem' }}>
                <h2 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.75rem' }}>Candidates</h2>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                        <tr style={{ borderBottom: '1px solid #e5e7eb' }}>
                            <th onClick={() => handleSort('candidate_id')} style={{ textAlign: 'left', padding: '0.5rem', fontWeight: 600, cursor: 'pointer' }}>ID {sortConfig.key === 'candidate_id' && (sortConfig.direction === 'asc' ? '↑' : '↓')}</th>
                            <th style={{ textAlign: 'left', padding: '0.5rem', fontWeight: 600 }}>Label</th>
                            <th style={{ textAlign: 'left', padding: '0.5rem', fontWeight: 600 }}>Preset</th>
                            <th onClick={() => handleSort('status')} style={{ textAlign: 'left', padding: '0.5rem', fontWeight: 600, cursor: 'pointer' }}>Status {sortConfig.key === 'status' && (sortConfig.direction === 'asc' ? '↑' : '↓')}</th>
                            <th style={{ textAlign: 'left', padding: '0.5rem', fontWeight: 600 }}>Run</th>
                            {scoreColumns.map(name => (
                                <th
                                    key={name}
                                    onClick={() => handleSort(`score:${name}`)}
                                    style={{ textAlign: 'left', padding: '0.5rem', fontWeight: 600, cursor: 'pointer', fontSize: '0.8rem' }}
                                >
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                                        {scoreLabels[name]}
                                        {scoreTypes[name] === 'proxy' && (
                                            <span style={{ fontSize: '0.7em', background: '#f3f4f6', border: '1px solid #d1d5db', padding: '0 3px', borderRadius: 3, color: '#6b7280', fontWeight: 500 }}>
                                                PROXY
                                            </span>
                                        )}
                                        {sortConfig.key === `score:${name}` && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                                    </div>
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {sortedRuns.map((run) => {
                            const candidate = experiment.candidates.find(c => c.candidate_id === run.candidate_id);
                            const colors = STATUS_COLORS[run.status] || { bg: '#f3f4f6', text: '#6b7280' };

                            return (
                                <tr key={run.candidate_id} style={{ borderBottom: '1px solid #e5e7eb' }}>
                                    <td style={{ padding: '0.5rem', fontWeight: 600 }}>{run.candidate_id}</td>
                                    <td style={{ padding: '0.5rem' }}>{candidate?.label || '-'}</td>
                                    <td style={{ padding: '0.5rem', fontFamily: 'monospace', fontSize: '0.875rem' }}>
                                        {run.steps_preset}
                                    </td>
                                    <td style={{ padding: '0.5rem' }}>
                                        <span style={{
                                            background: colors.bg,
                                            color: colors.text,
                                            padding: '0.25rem 0.5rem',
                                            borderRadius: 4,
                                            fontSize: '0.75rem',
                                            fontWeight: 600,
                                        }}>
                                            {run.status}
                                        </span>
                                    </td>
                                    <td style={{ padding: '0.5rem' }}>
                                        {run.run_id ? (
                                            <Link to={`/runs/${run.run_id}`} style={{ color: '#2563eb', textDecoration: 'underline' }}>
                                                {run.run_id.slice(0, 20)}...
                                            </Link>
                                        ) : (
                                            <span style={{ color: '#9ca3af' }}>—</span>
                                        )}
                                    </td>
                                    {scoreColumns.map(name => {
                                        const score = run.score_cards?.find(s => s.name === name)?.score;
                                        return (
                                            <td key={name} style={{ padding: '0.5rem' }}>
                                                {score !== undefined ? (
                                                    <span style={{
                                                        fontWeight: 600,
                                                        color: score >= 80 ? '#059669' : score >= 50 ? '#d97706' : '#dc2626'
                                                    }}>
                                                        {score}
                                                    </span>
                                                ) : <span style={{ color: '#d1d5db' }}>-</span>}
                                            </td>
                                        );
                                    })}
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>

            {/* Semantic Comparison Panel (B2) */}
            <div style={{ marginTop: '3rem', paddingTop: '2rem', borderTop: '1px solid #e5e7eb' }}>
                <h2 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '1.5rem' }}>Semantic Comparison</h2>

                {compLoading ? (
                    <div style={{ padding: '2rem', color: '#6b7280' }}>Analyzing results...</div>
                ) : compError ? (
                    <div style={{ padding: '1rem', background: '#fee2e2', color: '#dc2626', borderRadius: 6 }}>{compError}</div>
                ) : !comparison ? (
                    <div style={{ padding: '2rem', color: '#9ca3af', border: '2px dashed #e5e7eb', borderRadius: 8, textAlign: 'center' }}>
                        Comparison unavailable.
                    </div>
                ) : !comparison.readiness.comparable ? (
                    <div style={{ padding: '2rem', background: '#f9fafb', borderRadius: 8, textAlign: 'center' }}>
                        <div style={{ fontWeight: 600, color: '#4b5563', marginBottom: '0.5rem' }}>Evaluation Pending</div>
                        <div style={{ fontFamily: 'monospace', color: '#6b7280' }}>
                            Reason: {comparison.readiness.reason}
                        </div>
                    </div>
                ) : (
                    <div>
                        {/* Verdict Banner */}
                        <div style={{
                            padding: '1.5rem',
                            borderRadius: 8,
                            marginBottom: '2rem',
                            background: comparison.verdicts.overall.includes('BETTER') ? '#ecfdf5' : '#f3f4f6',
                            border: `1px solid ${comparison.verdicts.overall.includes('BETTER') ? '#a7f3d0' : '#e5e7eb'}`
                        }}>
                            <div style={{
                                fontWeight: 700, fontSize: '1.1rem', marginBottom: '0.5rem',
                                color: comparison.verdicts.overall.includes('BETTER') ? '#047857' : '#374151'
                            }}>
                                VERDICT: {comparison.verdicts.overall}
                            </div>
                            <ul style={{ margin: 0, paddingLeft: '1.2rem', color: '#4b5563' }}>
                                {comparison.verdicts.reasons.map((r, i) => (
                                    <li key={i}>{r}</li>
                                ))}
                            </ul>
                        </div>

                        {/* Metrics Table */}
                        <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '1rem' }}>Key Metrics</h3>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
                            <thead>
                                <tr style={{ background: '#f9fafb', borderBottom: '1px solid #e5e7eb' }}>
                                    <th style={{ textAlign: 'left', padding: '0.75rem' }}>Metric</th>
                                    <th style={{ textAlign: 'right', padding: '0.75rem' }}>
                                        {comparison.candidates.A.label} <span style={{ fontSize: '0.8em', color: '#9ca3af' }}>(A)</span>
                                    </th>
                                    <th style={{ textAlign: 'right', padding: '0.75rem' }}>
                                        {comparison.candidates.B.label} <span style={{ fontSize: '0.8em', color: '#9ca3af' }}>(B)</span>
                                    </th>
                                    <th style={{ textAlign: 'right', padding: '0.75rem' }}>Delta</th>
                                </tr>
                            </thead>
                            <tbody>
                                {comparison.metrics && Object.entries(comparison.metrics).map(([key, m]) => (
                                    <tr key={key} style={{ borderBottom: '1px solid #e5e7eb' }}>
                                        <td style={{ padding: '0.75rem', fontWeight: 500, color: '#374151' }}>{key}</td>
                                        <td style={{ padding: '0.75rem', textAlign: 'right', fontFamily: 'monospace' }}>
                                            {typeof m.A === 'number' ?
                                                (key === 'confidence_avg' ? (m.A * 100).toFixed(1) + '%' :
                                                    key === 'duration_s' ? m.A.toFixed(2) + 's' : m.A)
                                                : '-'}
                                        </td>
                                        <td style={{ padding: '0.75rem', textAlign: 'right', fontFamily: 'monospace' }}>
                                            {typeof m.B === 'number' ?
                                                (key === 'confidence_avg' ? (m.B * 100).toFixed(1) + '%' :
                                                    key === 'duration_s' ? m.B.toFixed(2) + 's' : m.B)
                                                : '-'}
                                        </td>
                                        <td style={{
                                            padding: '0.75rem', textAlign: 'right', fontFamily: 'monospace', fontWeight: 600,
                                            color: m.delta > 0 ? '#059669' : m.delta < 0 ? '#dc2626' : '#6b7280'
                                        }}>
                                            {m.pct_change > 0 ? '+' : ''}{m.pct_change.toFixed(1)}%
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>

            <div style={{ marginTop: '200px' }}></div>
        </div>
    );
}

