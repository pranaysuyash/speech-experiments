import { useState, useEffect, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import { api } from '../lib/api';

interface ExperimentRun {
    candidate_id: string;
    steps_preset: string;
    run_id: string | null;
    status: 'QUEUED' | 'STARTING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'STALE';
    created_at: string;
    started_at: string | null;
    ended_at: string | null;
    score_cards?: { name: string; label: string; score: number }[];
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

    // Compare state
    const [compareArtifact, setCompareArtifact] = useState<'transcript' | 'summary' | 'action_items'>('transcript');
    const [leftRunId, setLeftRunId] = useState<string>('');
    const [rightRunId, setRightRunId] = useState<string>('');
    const [compareData, setCompareData] = useState<any>(null);
    const [compareLoading, setCompareLoading] = useState(false);
    const [compareError, setCompareError] = useState<string | null>(null);

    // Sort state
    const [sortConfig, setSortConfig] = useState<{ key: string; direction: 'asc' | 'desc' }>({ key: 'candidate_id', direction: 'asc' });

    const didStartAll = useRef(false);

    // Initial load + start-all
    useEffect(() => {
        if (!experimentId) return;

        const loadAndStart = async () => {
            try {
                // Start-all once (best effort)
                if (!didStartAll.current) {
                    didStartAll.current = true;
                    await api.startExperimentAll(experimentId).catch(() => { });
                }

                const data = await api.getExperiment(experimentId);
                setExperiment(data);

                // key defaults off runs if available
                if (data.runs.length >= 2) {
                    const r1 = data.runs[0].run_id;
                    const r2 = data.runs[1].run_id;
                    if (r1) setLeftRunId(r1);
                    if (r2) setRightRunId(r2);
                } else if (data.runs.length === 1 && data.runs[0].run_id) {
                    setLeftRunId(data.runs[0].run_id);
                }
            } catch (e: any) {
                setError(e?.message || 'Failed to load experiment');
            }
        };

        loadAndStart();
    }, [experimentId]);

    // Polling + queue-lite
    useEffect(() => {
        if (!experimentId) return;

        const poll = async () => {
            try {
                const data = await api.getExperiment(experimentId);
                setExperiment(data);

                // Queue-lite: if any QUEUED and none RUNNING, try to start
                const hasQueued = data.runs.some((r: ExperimentRun) => r.status === 'QUEUED');
                const hasRunning = data.runs.some((r: ExperimentRun) => r.status === 'RUNNING');

                if (hasQueued && !hasRunning) {
                    await api.startExperimentNext(experimentId).catch(() => { });
                }
            } catch {
                // Ignore polling errors
            }
        };

        const interval = setInterval(poll, 1500);
        return () => clearInterval(interval);
    }, [experimentId]);

    // Fetch comparison
    useEffect(() => {
        if (!experimentId || !leftRunId || !rightRunId) return;

        const fetchCompare = async () => {
            setCompareLoading(true);
            setCompareError(null);
            try {
                const data = await api.getExperimentComparison(experimentId, leftRunId, rightRunId, compareArtifact);
                setCompareData(data);
            } catch (e: any) {
                // If 413, we still get data in body usually? 
                // Axios throws on 413. We need to check if response has data.
                if (e.response && e.response.status === 413) {
                    setCompareData(e.response.data); // Display the too large error state
                } else {
                    setCompareError('Failed to load comparison.');
                    setCompareData(null);
                }
            } finally {
                setCompareLoading(false);
            }
        };

        fetchCompare();
    }, [experimentId, leftRunId, rightRunId, compareArtifact]);

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
    experiment.runs.forEach(r => {
        r.score_cards?.forEach(s => {
            if (!scoreLabels[s.name]) scoreLabels[s.name] = s.label;
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
                                    {scoreLabels[name]} {sortConfig.key === `score:${name}` && (sortConfig.direction === 'asc' ? '↑' : '↓')}
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

            {/* Compare Panel */}
            <div style={{ marginTop: '3rem', paddingTop: '2rem', borderTop: '1px solid #e5e7eb' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                    <h2 style={{ fontSize: '1.25rem', fontWeight: 600 }}>Compare Runs</h2>

                    {/* Artifact Tabs */}
                    <div style={{ display: 'flex', background: '#f3f4f6', padding: '0.25rem', borderRadius: 8 }}>
                        {(['transcript', 'summary', 'action_items'] as const).map((art) => (
                            <button
                                key={art}
                                onClick={() => setCompareArtifact(art)}
                                style={{
                                    padding: '0.5rem 1rem',
                                    borderRadius: 6,
                                    fontSize: '0.875rem',
                                    fontWeight: 600,
                                    background: compareArtifact === art ? 'white' : 'transparent',
                                    color: compareArtifact === art ? '#111827' : '#6b7280',
                                    boxShadow: compareArtifact === art ? '0 1px 2px 0 rgba(0,0,0,0.05)' : 'none',
                                    cursor: 'pointer',
                                    border: 'none',
                                    textTransform: 'capitalize'
                                }}
                            >
                                {art.replace('_', ' ')}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Run Selectors */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
                    <select
                        value={leftRunId}
                        onChange={(e) => setLeftRunId(e.target.value)}
                        style={{ padding: '0.5rem', borderRadius: 6, border: '1px solid #d1d5db', width: '100%' }}
                    >
                        <option value="">Select Left Run...</option>
                        {experiment.runs.map(r => r.run_id && (
                            <option key={`L-${r.run_id}`} value={r.run_id}>
                                {experiment.candidates.find(c => c.candidate_id === r.candidate_id)?.label} ({r.run_id.slice(0, 8)}...)
                            </option>
                        ))}
                    </select>

                    <select
                        value={rightRunId}
                        onChange={(e) => setRightRunId(e.target.value)}
                        style={{ padding: '0.5rem', borderRadius: 6, border: '1px solid #d1d5db', width: '100%' }}
                    >
                        <option value="">Select Right Run...</option>
                        {experiment.runs.map(r => r.run_id && (
                            <option key={`R-${r.run_id}`} value={r.run_id}>
                                {experiment.candidates.find(c => c.candidate_id === r.candidate_id)?.label} ({r.run_id.slice(0, 8)}...)
                            </option>
                        ))}
                    </select>
                </div>

                {/* Comparison Content */}
                {compareLoading ? (
                    <div style={{ padding: '3rem', textAlign: 'center', color: '#6b7280', background: '#f9fafb', borderRadius: 8 }}>
                        Loading comparison...
                    </div>
                ) : compareError ? (
                    <div style={{ color: '#dc2626', background: '#fee2e2', padding: '1rem', borderRadius: 8 }}>
                        {compareError}
                    </div>
                ) : compareData ? (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                        {/* Left Content */}
                        <div style={{ border: '1px solid #e5e7eb', borderRadius: 8, overflow: 'hidden' }}>
                            <div style={{ background: '#f9fafb', padding: '0.5rem 1rem', borderBottom: '1px solid #e5e7eb', fontSize: '0.75rem', color: '#6b7280', display: 'flex', justifyContent: 'space-between' }}>
                                <span>{compareData.left?.size ? `${(compareData.left.size / 1024).toFixed(1)} KB` : '0 KB'}</span>
                                {compareData.left?.truncated && <span style={{ color: '#d97706', fontWeight: 600 }}>TRUNCATED</span>}
                            </div>
                            <pre style={{ margin: 0, padding: '1rem', fontSize: '0.8rem', whiteSpace: 'pre-wrap', maxHeight: '600px', overflowY: 'auto', background: compareData.left?.available ? 'white' : '#f3f4f6', color: compareData.left?.available ? 'inherit' : '#9ca3af' }}>
                                {compareData.left?.error ? `Error: ${compareData.left.error}` :
                                    compareData.left?.available ? compareData.left.text : 'Artifact not available'}
                            </pre>
                        </div>

                        {/* Right Content */}
                        <div style={{ border: '1px solid #e5e7eb', borderRadius: 8, overflow: 'hidden' }}>
                            <div style={{ background: '#f9fafb', padding: '0.5rem 1rem', borderBottom: '1px solid #e5e7eb', fontSize: '0.75rem', color: '#6b7280', display: 'flex', justifyContent: 'space-between' }}>
                                <span>{compareData.right?.size ? `${(compareData.right.size / 1024).toFixed(1)} KB` : '0 KB'}</span>
                                {compareData.right?.truncated && <span style={{ color: '#d97706', fontWeight: 600 }}>TRUNCATED</span>}
                            </div>
                            <pre style={{ margin: 0, padding: '1rem', fontSize: '0.8rem', whiteSpace: 'pre-wrap', maxHeight: '600px', overflowY: 'auto', background: compareData.right?.available ? 'white' : '#f3f4f6', color: compareData.right?.available ? 'inherit' : '#9ca3af' }}>
                                {compareData.right?.error ? `Error: ${compareData.right.error}` :
                                    compareData.right?.available ? compareData.right.text : 'Artifact not available'}
                            </pre>
                        </div>
                    </div>
                ) : (
                    <div style={{ padding: '3rem', textAlign: 'center', color: '#9ca3af', border: '2px dashed #e5e7eb', borderRadius: 8 }}>
                        Select two runs to compare.
                    </div>
                )}
            </div>

            <div style={{ marginTop: '200px' }}></div>
        </div>
    );
}

