import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';

type WorkbenchResponse =
  | { run_id: string; run_dir: string; console_url: string }
  | { error_code: string };

export default function WorkbenchPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [useCaseId, setUseCaseId] = useState('meeting_smoke');
  const [stepsPreset, setStepsPreset] = useState<'full' | 'ingest'>('ingest');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const useCases = useMemo(
    () => [
      { id: 'meeting_smoke', name: 'Meeting Smoke' },
      { id: 'asr_smoke', name: 'ASR Smoke' },
    ],
    [],
  );

  async function onStart() {
    if (!file || isSubmitting) return;
    setError(null);
    setIsSubmitting(true);
    try {
      const form = new FormData();
      form.append('file', file);
      form.append('use_case_id', useCaseId);
      form.append('steps_preset', stepsPreset);

      const res = await fetch('/api/workbench/runs', { method: 'POST', body: form });
      const payload = (await res.json()) as WorkbenchResponse;

      if (!res.ok) {
        if ('error_code' in payload && payload.error_code === 'RUNNER_BUSY') {
          setError('Runner is busy. Wait for the current run to finish, then try again.');
          return;
        }
        setError(`Failed to start run (${res.status}).`);
        return;
      }

      if (!('run_id' in payload)) {
        setError('Unexpected response from server.');
        return;
      }

      navigate(`/runs/${payload.run_id}`);
    } catch (e: any) {
      setError(e?.message || 'Failed to start run.');
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div style={{ padding: '2rem' }}>
      <h1 style={{ fontSize: '1.4rem', fontWeight: 700, marginBottom: '1rem' }}>Workbench</h1>

      <div style={{ display: 'grid', gap: '1rem', maxWidth: 720 }}>
        <label style={{ display: 'grid', gap: '0.25rem' }}>
          <div style={{ fontWeight: 600 }}>Use Case</div>
          <select
            value={useCaseId}
            onChange={(e) => setUseCaseId(e.target.value)}
            style={{ padding: '0.5rem', border: '1px solid #e5e7eb', borderRadius: 6 }}
          >
            {useCases.map((uc) => (
              <option key={uc.id} value={uc.id}>
                {uc.name} ({uc.id})
              </option>
            ))}
          </select>
        </label>

        <label style={{ display: 'grid', gap: '0.25rem' }}>
          <div style={{ fontWeight: 600 }}>Input File</div>
          <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
        </label>

        <label style={{ display: 'grid', gap: '0.25rem' }}>
          <div style={{ fontWeight: 600 }}>Steps Preset</div>
          <select
            value={stepsPreset}
            onChange={(e) => setStepsPreset(e.target.value as 'full' | 'ingest')}
            style={{ padding: '0.5rem', border: '1px solid #e5e7eb', borderRadius: 6 }}
          >
            <option value="ingest">Ingest-only</option>
            <option value="full">Full pipeline</option>
          </select>
        </label>

        {error ? (
          <div style={{ color: '#b91c1c', background: '#fef2f2', padding: '0.75rem', borderRadius: 6 }}>
            {error}
          </div>
        ) : null}

        <button
          onClick={onStart}
          disabled={!file || isSubmitting}
          style={{
            padding: '0.75rem 1rem',
            borderRadius: 8,
            border: '1px solid #111827',
            background: !file || isSubmitting ? '#e5e7eb' : '#111827',
            color: !file || isSubmitting ? '#6b7280' : 'white',
            fontWeight: 700,
            cursor: !file || isSubmitting ? 'not-allowed' : 'pointer',
          }}
        >
          {isSubmitting ? 'Starting…' : 'Start Run'}
        </button>
      </div>
    </div>
  );
}

