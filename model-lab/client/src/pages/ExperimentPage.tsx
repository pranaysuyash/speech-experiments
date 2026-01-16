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
    const [previewA, setPreviewA] = useState<string | null>(null);
    const [previewB, setPreviewB] = useState<string | null>(null);
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

    // Load previews when runs complete
    useEffect(() => {
        if (!experiment) return;

        const loadPreview = async (run: ExperimentRun, setPreview: (s: string | null) => void) => {
            if (!run.run_id || run.status !== 'COMPLETED') {
                setPreview(null);
                return;
            }

            try {
                // Try transcript.txt first
                const url = api.getMeetingPackArtifactPreviewUrl(run.run_id, 'transcript.txt', 5000);
                const res = await fetch(url);
                if (res.ok) {
                    setPreview(await res.text());
                } else {
                    setPreview('Not produced by this preset.');
                }
            } catch {
                setPreview('Preview unavailable.');
            }
        };

        const runA = experiment.runs.find(r => r.candidate_id === 'A');
        const runB = experiment.runs.find(r => r.candidate_id === 'B');

        if (runA) loadPreview(runA, setPreviewA);
        if (runB) loadPreview(runB, setPreviewB);
    }, [experiment?.runs]);

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
                            <th style={{ textAlign: 'left', padding: '0.5rem', fontWeight: 600 }}>ID</th>
                            <th style={{ textAlign: 'left', padding: '0.5rem', fontWeight: 600 }}>Label</th>
                            <th style={{ textAlign: 'left', padding: '0.5rem', fontWeight: 600 }}>Preset</th>
                            <th style={{ textAlign: 'left', padding: '0.5rem', fontWeight: 600 }}>Status</th>
                            <th style={{ textAlign: 'left', padding: '0.5rem', fontWeight: 600 }}>Run</th>
                        </tr>
                    </thead>
                    <tbody>
                        {experiment.runs.map((run) => {
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
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>

            {/* Compare Panel */}
            <div>
                <h2 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.75rem' }}>Compare</h2>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                    {/* Candidate A */}
                    <div style={{ border: '1px solid #e5e7eb', borderRadius: 8, padding: '1rem' }}>
                        <h3 style={{ fontWeight: 600, marginBottom: '0.5rem' }}>Candidate A</h3>
                        {experiment.runs.find(r => r.candidate_id === 'A')?.run_id ? (
                            previewA ? (
                                <pre style={{
                                    whiteSpace: 'pre-wrap',
                                    fontSize: '0.75rem',
                                    background: '#f9fafb',
                                    padding: '0.75rem',
                                    borderRadius: 4,
                                    maxHeight: 300,
                                    overflow: 'auto'
                                }}>
                                    {previewA}
                                </pre>
                            ) : (
                                <div style={{ color: '#6b7280', fontStyle: 'italic' }}>Loading preview...</div>
                            )
                        ) : (
                            <div style={{ color: '#6b7280', fontStyle: 'italic' }}>Waiting for run to start...</div>
                        )}
                    </div>

                    {/* Candidate B */}
                    <div style={{ border: '1px solid #e5e7eb', borderRadius: 8, padding: '1rem' }}>
                        <h3 style={{ fontWeight: 600, marginBottom: '0.5rem' }}>Candidate B</h3>
                        {experiment.runs.find(r => r.candidate_id === 'B')?.run_id ? (
                            previewB ? (
                                <pre style={{
                                    whiteSpace: 'pre-wrap',
                                    fontSize: '0.75rem',
                                    background: '#f9fafb',
                                    padding: '0.75rem',
                                    borderRadius: 4,
                                    maxHeight: 300,
                                    overflow: 'auto'
                                }}>
                                    {previewB}
                                </pre>
                            ) : (
                                <div style={{ color: '#6b7280', fontStyle: 'italic' }}>Loading preview...</div>
                            )
                        ) : (
                            <div style={{ color: '#6b7280', fontStyle: 'italic' }}>Waiting for run to start...</div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
