import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../lib/api';

const ACCEPTED_FORMATS = 'audio/*, video/*, .wav, .mp3, .m4a, .mp4, .mov, .avi';
const SIZE_WARNING_BYTES = 500 * 1024 * 1024; // 500MB

type Mode = 'single' | 'compare';

interface Candidate {
  candidate_id: string;
  label: string;
  use_case_id: string;
  steps_preset: string;
}

interface UseCase {
  use_case_id: string;
  title: string;
}

export default function WorkbenchPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);

  // Selections
  const [mode, setMode] = useState<Mode>('single');
  const [useCaseId, setUseCaseId] = useState('');
  const [candidateId, setCandidateId] = useState('');     // Single
  const [candidateIdA, setCandidateIdA] = useState('');    // Compare
  const [candidateIdB, setCandidateIdB] = useState('');    // Compare

  // Data
  const [useCases, setUseCases] = useState<UseCase[]>([]);
  const [candidates, setCandidates] = useState<Candidate[]>([]);

  // State
  const [loadingConfig, setLoadingConfig] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 1. Initial Load: Use Cases
  useEffect(() => {
    (async () => {
      try {
        const ucs = await api.getUseCases();
        setUseCases(ucs);
        if (ucs.length > 0) setUseCaseId(ucs[0].use_case_id);
      } catch (e) {
        setError("Failed to load use cases");
      } finally {
        setLoadingConfig(false);
      }
    })();
  }, []);

  // 2. Load Candidates when Use Case changes
  useEffect(() => {
    if (!useCaseId) return;
    (async () => {
      try {
        const cands = await api.getCandidatesForUseCase(useCaseId);
        setCandidates(cands);

        // Default selections
        if (cands.length > 0) {
          setCandidateId(cands[0].candidate_id);
          setCandidateIdA(cands[0].candidate_id);
          setCandidateIdB(cands.length > 1 ? cands[1].candidate_id : cands[0].candidate_id);
        } else {
          setCandidateId('');
          setCandidateIdA('');
          setCandidateIdB('');
        }
      } catch (e) {
        console.error("Failed to load candidates", e);
      }
    })();
  }, [useCaseId]);

  const sizeWarning = file && file.size > SIZE_WARNING_BYTES
    ? `Warning: Large file (${(file.size / (1024 * 1024)).toFixed(0)}MB). Upload may take time.`
    : null;

  async function onStart() {
    if (!file || isSubmitting) return;
    setError(null);
    setIsSubmitting(true);

    try {
      const candsToRun = mode === 'single' ? [candidateId] : [candidateIdA, candidateIdB];

      // Runtime Invariants (Option A)
      if (mode === 'single' && (candsToRun.length !== 1 || !candsToRun[0])) {
        throw new Error("Invariant Violation: Single mode requires exactly 1 selected candidate.");
      }
      if (mode === 'compare' && (candsToRun.length !== 2 || !candsToRun[0] || !candsToRun[1])) {
        throw new Error("Invariant Violation: Compare mode requires exactly 2 selected candidates.");
      }

      // 1. Create Experiment (The container for everything)
      // Note: backend expects candidate_ids mainly for snapshotting
      const exp = await api.createExperiment(file, useCaseId, candsToRun);
      const expId = exp.experiment_id;

      if (mode === 'single') {
        // 2. Single: Start one specific candidate run
        const result = await api.startExperimentNext(expId);

        // 3. Redirect directly to the Run Detail
        if (result && result.run_id) {
          navigate(`/runs/${result.run_id}`);
        } else {
          navigate(`/lab/experiments/${expId}`);
        }
      } else {
        // 2. Compare: Start all (A and B)
        await api.startExperimentAll(expId);

        // 3. Redirect to Experiment Detail
        navigate(`/lab/experiments/${expId}`);
      }

    } catch (e: any) {
      const data = e?.response?.data;
      const msg = data?.error_message || data?.detail || e?.message || 'Failed to start.';
      setError(msg);
      // Optional: Log trace if available
      if (data?.trace_id) console.error("Trace:", data.trace_id);
    } finally {
      setIsSubmitting(false);
    }
  }

  if (loadingConfig) return <div className="p-8">Loading workbench...</div>;

  return (
    <div className="p-8 max-w-3xl">
      <h1 className="text-2xl font-bold mb-6">Workbench</h1>

      <div className="space-y-6 bg-white p-6 rounded-lg border shadow-sm">

        {/* Row 1: Use Case & Mode */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <label className="block">
            <span className="block font-semibold mb-1">Use Case</span>
            <select
              value={useCaseId}
              onChange={(e) => setUseCaseId(e.target.value)}
              disabled={isSubmitting}
              className="w-full border p-2 rounded"
            >
              {useCases.map(u => (
                <option key={u.use_case_id} value={u.use_case_id}>{u.title}</option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className="block font-semibold mb-1">Mode</span>
            <div className="flex rounded bg-gray-100 p-1">
              <button
                className={`flex-1 py-1 px-3 rounded text-sm font-medium transition-colors ${mode === 'single' ? 'bg-white shadow text-gray-900' : 'text-gray-500 hover:text-gray-900'}`}
                onClick={() => setMode('single')}
                disabled={isSubmitting}
              >
                Single Run
              </button>
              <button
                className={`flex-1 py-1 px-3 rounded text-sm font-medium transition-colors ${mode === 'compare' ? 'bg-white shadow text-blue-600' : 'text-gray-500 hover:text-gray-900'}`}
                onClick={() => setMode('compare')}
                disabled={isSubmitting}
              >
                Compare
              </button>
            </div>
          </label>
        </div>

        {/* Row 2: File */}
        <label className="block">
          <span className="block font-semibold mb-1">Input File</span>
          <input
            type="file"
            accept={ACCEPTED_FORMATS}
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            disabled={isSubmitting}
            className="w-full border p-2 rounded"
          />
          <div className="text-xs text-gray-500 mt-1">
            Accepted: Audio/Video files (WAV, MP3, M4A, MP4, MOV, etc.)
          </div>
        </label>

        {sizeWarning && (
          <div className="text-amber-800 bg-amber-50 p-3 rounded text-sm">
            {sizeWarning}
          </div>
        )}

        {/* Row 3: Candidate Selection */}
        <div className="border-t pt-4">
          {mode === 'single' ? (
            <label className="block">
              <span className="block font-semibold mb-1">Candidate</span>
              <select
                value={candidateId}
                onChange={(e) => setCandidateId(e.target.value)}
                disabled={isSubmitting}
                className="w-full border p-2 rounded bg-gray-50 font-mono text-sm"
              >
                {candidates.map(c => (
                  <option key={c.candidate_id} value={c.candidate_id}>
                    {c.label} ({c.candidate_id}) [{c.steps_preset}]
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <div className="grid grid-cols-2 gap-4">
              <label className="block">
                <span className="block font-semibold mb-1 text-blue-600">Candidate A</span>
                <select
                  value={candidateIdA}
                  onChange={(e) => setCandidateIdA(e.target.value)}
                  disabled={isSubmitting}
                  className="w-full border p-2 rounded bg-gray-50 font-mono text-xs"
                >
                  {candidates.map(c => (
                    <option key={c.candidate_id} value={c.candidate_id}>
                      {c.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="block">
                <span className="block font-semibold mb-1 text-purple-600">Candidate B</span>
                <select
                  value={candidateIdB}
                  onChange={(e) => setCandidateIdB(e.target.value)}
                  disabled={isSubmitting}
                  className="w-full border p-2 rounded bg-gray-50 font-mono text-xs"
                >
                  {candidates.map(c => (
                    <option key={c.candidate_id} value={c.candidate_id}>
                      {c.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          )}
        </div>

        {error && (
          <div className="text-red-800 bg-red-50 p-3 rounded text-sm font-medium">
            {error}
          </div>
        )}

        <button
          onClick={onStart}
          disabled={!file || isSubmitting || (mode === 'single' && !candidateId) || (mode === 'compare' && (!candidateIdA || !candidateIdB))}
          className={`w-full py-3 rounded font-bold text-white transition-colors ${isSubmitting ? 'bg-gray-400 cursor-not-allowed' : 'bg-gray-900 hover:bg-black'
            }`}
        >
          {isSubmitting ? 'Initializing...' : (mode === 'single' ? 'Start Run' : 'Start Comparison')}
        </button>

      </div>
    </div>
  );
}
