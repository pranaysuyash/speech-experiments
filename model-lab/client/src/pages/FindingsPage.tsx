import { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE = '/api';

interface Finding {
    finding_id: string;
    title: string;
    category: string;
    severity: string;
    details: string;
    count: number;
    first_seen_at: string;
    last_seen_at: string;
    latest_run_id: string;
    evidence_paths: string[];
}

export default function FindingsPage() {
    const [findings, setFindings] = useState<Finding[]>([]);
    const [loading, setLoading] = useState(true);
    const [filters, setFilters] = useState({
        severity: '',
        category: ''
    });

    useEffect(() => {
        fetchFindings();
    }, []);

    const fetchFindings = async () => {
        setLoading(true);
        try {
            const params = new URLSearchParams();
            if (filters.severity) params.append('severity', filters.severity);
            if (filters.category) params.append('category', filters.category);

            const response = await axios.get(`${API_BASE}/findings?${params}`);
            setFindings(response.data);
        } catch (error) {
            console.error('Failed to fetch findings:', error);
        } finally {
            setLoading(false);
        }
    };

    const severityColor = (severity: string) => {
        switch (severity) {
            case 'high': return { bg: '#fee2e2', text: '#991b1b' };
            case 'medium': return { bg: '#fef3c7', text: '#92400e' };
            case 'low': return { bg: '#dbeafe', text: '#1e40af' };
            default: return { bg: '#f3f4f6', text: '#374151' };
        }
    };

    return (
        <div style={{ padding: '2rem', maxWidth: '1400px', margin: '0 auto' }}>
            <h1>Lab Findings</h1>
            <p style={{ color: '#6b7280', marginBottom: '2rem' }}>
                Aggregated issues and observations across all runs
            </p>

            {/* Filters */}
            <div style={{
                display: 'flex',
                gap: '1rem',
                marginBottom: '2rem',
                padding: '1rem',
                background: '#f5f5f5',
                borderRadius: '8px'
            }}>
                <select
                    value={filters.severity}
                    onChange={(e) => setFilters(prev => ({ ...prev, severity: e.target.value }))}
                    style={{ padding: '0.5rem', flex: 1 }}
                >
                    <option value="">All Severities</option>
                    <option value="high">High</option>
                    <option value="medium">Medium</option>
                    <option value="low">Low</option>
                </select>
                <select
                    value={filters.category}
                    onChange={(e) => setFilters(prev => ({ ...prev, category: e.target.value }))}
                    style={{ padding: '0.5rem', flex: 1 }}
                >
                    <option value="">All Categories</option>
                    <option value="asr">ASR</option>
                    <option value="diarization">Diarization</option>
                    <option value="alignment">Alignment</option>
                    <option value="system">System</option>
                </select>
                <button onClick={fetchFindings} style={{ padding: '0.5rem 1.5rem' }}>
                    Apply Filters
                </button>
            </div>

            {/* Findings List */}
            {loading ? (
                <div>Loading findings...</div>
            ) : findings.length === 0 ? (
                <div style={{
                    padding: '3rem',
                    textAlign: 'center',
                    background: 'white',
                    borderRadius: '8px',
                    boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
                }}>
                    <p style={{ color: '#9ca3af', fontSize: '1.1em' }}>No findings found</p>
                </div>
            ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    {findings.map((finding) => {
                        const colors = severityColor(finding.severity);
                        return (
                            <div
                                key={finding.finding_id}
                                style={{
                                    background: 'white',
                                    padding: '1.5rem',
                                    borderRadius: '8px',
                                    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                                    borderLeft: `4px solid ${colors.text}`
                                }}
                            >
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '1rem' }}>
                                    <div>
                                        <h3 style={{ margin: 0, marginBottom: '0.5rem' }}>{finding.title}</h3>
                                        <div style={{ display: 'flex', gap: '0.75rem', fontSize: '0.9em' }}>
                                            <span style={{
                                                padding: '0.25rem 0.75rem',
                                                borderRadius: '12px',
                                                background: colors.bg,
                                                color: colors.text,
                                                fontWeight: 500
                                            }}>
                                                {finding.severity.toUpperCase()}
                                            </span>
                                            <span style={{ color: '#6b7280' }}>
                                                {finding.category}
                                            </span>
                                        </div>
                                    </div>
                                    <div style={{ textAlign: 'right' }}>
                                        <div style={{ fontSize: '1.5em', fontWeight: 'bold', color: '#374151' }}>
                                            {finding.count}
                                        </div>
                                        <div style={{ fontSize: '0.85em', color: '#9ca3af' }}>
                                            occurrence{finding.count !== 1 ? 's' : ''}
                                        </div>
                                    </div>
                                </div>

                                <p style={{ color: '#4b5563', margin: '0 0 1rem 0' }}>
                                    {finding.details}
                                </p>

                                <div style={{
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'center',
                                    paddingTop: '1rem',
                                    borderTop: '1px solid #e5e7eb',
                                    fontSize: '0.9em'
                                }}>
                                    <div style={{ color: '#6b7280' }}>
                                        First seen: {new Date(finding.first_seen_at).toLocaleDateString()} •
                                        Last seen: {new Date(finding.last_seen_at).toLocaleDateString()}
                                    </div>
                                    <a
                                        href={`/runs/${finding.latest_run_id}`}
                                        style={{
                                            color: '#2563eb',
                                            textDecoration: 'none',
                                            fontWeight: 500
                                        }}
                                    >
                                        View Latest Run →
                                    </a>
                                </div>

                                {finding.evidence_paths.length > 0 && (
                                    <div style={{ marginTop: '1rem', fontSize: '0.85em' }}>
                                        <strong>Evidence:</strong>
                                        <ul style={{ margin: '0.5rem 0 0 0', paddingLeft: '1.5rem', color: '#6b7280' }}>
                                            {finding.evidence_paths.slice(0, 3).map((path, idx) => (
                                                <li key={idx} style={{ fontFamily: 'monospace' }}>{path}</li>
                                            ))}
                                            {finding.evidence_paths.length > 3 && (
                                                <li>... and {finding.evidence_paths.length - 3} more</li>
                                            )}
                                        </ul>
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
