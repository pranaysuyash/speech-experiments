import { useMemo, useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../lib/api';

type WorkbenchResponse =
  | { run_id: string; run_dir: string; console_url: string }
  | { error_code: string };

const ACCEPTED_FORMATS = 'audio/*, video/*, .wav, .mp3, .m4a, .mp4, .mov, .avi';
const SIZE_WARNING_BYTES = 500 * 1024 * 1024; // 500MB

type Mode = 'single' | 'compare';

export default function WorkbenchPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [useCaseId, setUseCaseId] = useState('meeting_smoke');
  const [stepsPreset, setStepsPreset] = useState<'full' | 'ingest'>('ingest');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [retryCountdown, setRetryCountdown] = useState(0);
  const [mode, setMode] = useState<Mode>('single');

  const useCases = useMemo(
    () => [
      { id: 'meeting_smoke', name: 'Meeting Smoke' },
      { id: 'asr_smoke', name: 'ASR Smoke' },
    ],
    [],
  );

  // Countdown for retry
  useEffect(() => {
    if (retryCountdown > 0) {
      const timer = setTimeout(() => setRetryCountdown(retryCountdown - 1), 1000);
      return () => clearTimeout(timer);
    }
  }, [retryCountdown]);

  const sizeWarning = file && file.size > SIZE_WARNING_BYTES
    ? `Warning: Large file (${(file.size / (1024 * 1024)).toFixed(0)}MB). Upload may take time.`
    : null;

  async function onStart() {
    if (!file || isSubmitting) return;
    setError(null);
    setIsBusy(false);
    setIsSubmitting(true);

    try {
      // Compare mode: create experiment and redirect
      if (mode === 'compare') {
        const result = await api.createExperiment(file, useCaseId);
        navigate(`/lab/experiments/${result.experiment_id}`);
        return;
      }

      // Single mode: existing workflow
      const form = new FormData();
      form.append('file', file);
      form.append('use_case_id', useCaseId);
      form.append('steps_preset', stepsPreset);

      const res = await fetch('/api/workbench/runs', { method: 'POST', body: form });
      const payload = (await res.json()) as WorkbenchResponse;

      if (!res.ok) {
        if ('error_code' in payload && payload.error_code === 'RUNNER_BUSY') {
          setIsBusy(true);
          setRetryCountdown(10);
          setError('Runner is busy with another session. Please wait and try again.');
          return;
        }
        setError(`Failed to start run (${res.status}): ${JSON.stringify(payload)}`);
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
            disabled={isSubmitting}
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
          <div style={{ fontWeight: 600 }}>Mode</div>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              type="button"
              onClick={() => setMode('single')}
              disabled={isSubmitting}
              style={{
                flex: 1,
                padding: '0.5rem',
                borderRadius: 6,
                border: mode === 'single' ? '2px solid #111827' : '1px solid #e5e7eb',
                background: mode === 'single' ? '#111827' : 'white',
                color: mode === 'single' ? 'white' : '#374151',
                fontWeight: 600,
                cursor: isSubmitting ? 'not-allowed' : 'pointer',
              }}
            >
              Single Run
            </button>
            <button
              type="button"
              onClick={() => setMode('compare')}
              disabled={isSubmitting}
              style={{
                flex: 1,
                padding: '0.5rem',
                borderRadius: 6,
                border: mode === 'compare' ? '2px solid #2563eb' : '1px solid #e5e7eb',
                background: mode === 'compare' ? '#2563eb' : 'white',
                color: mode === 'compare' ? 'white' : '#374151',
                fontWeight: 600,
                cursor: isSubmitting ? 'not-allowed' : 'pointer',
              }}
            >
              Compare (2 runs)
            </button>
          </div>
          {mode === 'compare' && (
            <div style={{ fontSize: '0.75rem', color: '#6b7280', marginTop: 4 }}>
              Runs both "Ingest Only" and "Full Pipeline" on the same file
            </div>
          )}
        </label>

        <label style={{ display: 'grid', gap: '0.25rem' }}>
          <div style={{ fontWeight: 600 }}>Input File</div>
          <input
            type="file"
            accept={ACCEPTED_FORMATS}
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            disabled={isSubmitting}
          />
          <div style={{ fontSize: '0.75rem', color: '#6b7280', marginTop: 4 }}>
            Accepted: Audio/Video files (WAV, MP3, M4A, MP4, MOV, AVI, etc.)
          </div>
        </label>

        <label style={{ display: 'grid', gap: '0.25rem' }}>
          <div style={{ fontWeight: 600 }}>Steps Preset</div>
          <select
            value={stepsPreset}
            onChange={(e) => setStepsPreset(e.target.value as 'full' | 'ingest')}
            disabled={isSubmitting}
            style={{ padding: '0.5rem', border: '1px solid #e5e7eb', borderRadius: 6 }}
          >
            <option value="ingest">Ingest-only</option>
            <option value="full">Full pipeline</option>
          </select>
        </label>

        {sizeWarning && (
          <div style={{ color: '#d97706', background: '#fffbeb', padding: '0.75rem', borderRadius: 6, fontSize: '0.875rem' }}>
            {sizeWarning}
          </div>
        )}

        {error && (
          <div style={{ color: '#b91c1c', background: '#fef2f2', padding: '0.75rem', borderRadius: 6 }}>
            {error}
            {isBusy && (
              <div style={{ marginTop: '0.5rem', fontSize: '0.875rem', color: '#6b7280' }}>
                {retryCountdown > 0
                  ? `You can retry in ${retryCountdown} seconds...`
                  : 'You can retry now.'}
              </div>
            )}
          </div>
        )}

        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            onClick={onStart}
            disabled={!file || isSubmitting}
            style={{
              flex: 1,
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

          {isBusy && retryCountdown === 0 && (
            <button
              onClick={onStart}
              disabled={!file}
              style={{
                padding: '0.75rem 1rem',
                borderRadius: 8,
                border: '1px solid #d97706',
                background: '#fffbeb',
                color: '#d97706',
                fontWeight: 700,
                cursor: !file ? 'not-allowed' : 'pointer',
              }}
            >
              Retry Now
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

