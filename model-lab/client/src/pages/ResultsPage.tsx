import { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE = '/api';

interface ResultSummary {
    run_id: string;
    status: string;
    started_at?: string;
    duration_ms?: number;
    steps_completed: string[];
    eval_available: boolean;
    use_case_id?: string | null;
    model_id?: string | null;
    metrics: Record<string, number>;
    checks_passed?: number | null;
    checks_total?: number | null;
}

export default function ResultsPage() {
    const [results, setResults] = useState<ResultSummary[]>([]);
    const [loading, setLoading] = useState(true);
    const [filters, setFilters] = useState({
        useCase: '',
        model: '',
        status: ''
    });

    useEffect(() => {
        fetchResults();
    }, []);

    const fetchResults = async () => {
        setLoading(true);
        try {
            const params = new URLSearchParams();
            if (filters.useCase) params.append('use_case_id', filters.useCase);
            if (filters.model) params.append('model_id', filters.model);
            if (filters.status) params.append('status', filters.status);

            const response = await axios.get(`${API_BASE}/results?${params}`);
            setResults(response.data);
        } catch (error) {
            console.error('Failed to fetch results:', error);
        } finally {
            setLoading(false);
        }
    };

    const formatMetrics = (metrics: Record<string, number>) => {
        const entries = Object.entries(metrics);
        if (entries.length === 0) return '-';
        return entries
            .slice(0, 3)  // Show first 3 metrics
            .map(([key, val]) => `${key}: ${(val * 100).toFixed(1)}%`)
            .join(', ');
    };

    return (
        <div style={{ padding: '2rem', maxWidth: '1400px', margin: '0 auto' }}>
            <h1>Lab Results</h1>

            {/* Filters */}
            <div style={{
                display: 'flex',
                gap: '1rem',
                marginBottom: '2rem',
                padding: '1rem',
                background: '#f5f5f5',
                borderRadius: '8px'
            }}>
                <input
                    type="text"
                    placeholder="Use Case ID"
                    value={filters.useCase}
                    onChange={(e) => setFilters(prev => ({ ...prev, useCase: e.target.value }))}
                    style={{ padding: '0.5rem', flex: 1 }}
                />
                <input
                    type="text"
                    placeholder="Model ID"
                    value={filters.model}
                    onChange={(e) => setFilters(prev => ({ ...prev, model: e.target.value }))}
                    style={{ padding: '0.5rem', flex: 1 }}
                />
                <select
                    value={filters.status}
                    onChange={(e) => setFilters(prev => ({ ...prev, status: e.target.value }))}
                    style={{ padding: '0.5rem', flex: 1 }}
                >
                    <option value="">All Statuses</option>
                    <option value="COMPLETED">COMPLETED</option>
                    <option value="RUNNING">RUNNING</option>
                    <option value="FAILED">FAILED</option>
                </select>
                <button onClick={fetchResults} style={{ padding: '0.5rem 1.5rem' }}>
                    Apply Filters
                </button>
            </div>

            {/* Results Table */}
            {loading ? (
                <div>Loading results...</div>
            ) : (
                <div style={{ overflowX: 'auto' }}>
                    <table style={{
                        width: '100%',
                        borderCollapse: 'collapse',
                        background: 'white',
                        boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
                    }}>
                        <thead>
                            <tr style={{ background: '#f9fafb', borderBottom: '2px solid #e5e7eb' }}>
                                <th style={{ padding: '1rem', textAlign: 'left' }}>Run ID</th>
                                <th style={{ padding: '1rem', textAlign: 'left' }}>Status</th>
                                <th style={{ padding: '1rem', textAlign: 'left' }}>Model</th>
                                <th style={{ padding: '1rem', textAlign: 'left' }}>Use Case</th>
                                <th style={{ padding: '1rem', textAlign: 'left' }}>Key Metrics</th>
                                <th style={{ padding: '1rem', textAlign: 'left' }}>Checks</th>
                                <th style={{ padding: '1rem', textAlign: 'left' }}>Started</th>
                            </tr>
                        </thead>
                        <tbody>
                            {results.map((result) => (
                                <tr
                                    key={result.run_id}
                                    onClick={() => window.location.href = `/runs/${result.run_id}`}
                                    style={{
                                        borderBottom: '1px solid #e5e7eb',
                                        cursor: 'pointer'
                                    }}
                                    onMouseEnter={(e) => e.currentTarget.style.background = '#f9fafb'}
                                    onMouseLeave={(e) => e.currentTarget.style.background = 'white'}
                                >
                                    <td style={{ padding: '1rem', fontFamily: 'monospace', fontSize: '0.9em' }}>
                                        {result.run_id.slice(0, 20)}...
                                    </td>
                                    <td style={{ padding: '1rem' }}>
                                        <span style={{
                                            padding: '0.25rem 0.75rem',
                                            borderRadius: '12px',
                                            fontSize: '0.85em',
                                            fontWeight: 500,
                                            background: result.status === 'COMPLETED' ? '#d1fae5' :
                                                result.status === 'FAILED' ? '#fee2e2' : '#fef3c7',
                                            color: result.status === 'COMPLETED' ? '#065f46' :
                                                result.status === 'FAILED' ? '#991b1b' : '#92400e'
                                        }}>
                                            {result.status}
                                        </span>
                                    </td>
                                    <td style={{ padding: '1rem' }}>{result.model_id || '-'}</td>
                                    <td style={{ padding: '1rem' }}>{result.use_case_id || '-'}</td>
                                    <td style={{ padding: '1rem', fontSize: '0.9em' }}>
                                        {result.eval_available ? formatMetrics(result.metrics) : 'No eval'}
                                    </td>
                                    <td style={{ padding: '1rem' }}>
                                        {result.checks_total !== null ? (
                                            <span style={{ color: result.checks_passed === result.checks_total ? '#059669' : '#dc2626' }}>
                                                {result.checks_passed}/{result.checks_total}
                                            </span>
                                        ) : '-'}
                                    </td>
                                    <td style={{ padding: '1rem', fontSize: '0.9em', color: '#6b7280' }}>
                                        {result.started_at ? new Date(result.started_at).toLocaleString() : '-'}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>

                    {results.length === 0 && (
                        <div style={{ padding: '3rem', textAlign: 'center', color: '#9ca3af' }}>
                            No results found matching filters
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
