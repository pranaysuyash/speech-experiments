import { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../lib/api';

const ACCEPTED_FORMATS = 'audio/*, video/*, .wav, .mp3, .m4a, .mp4, .mov, .avi';
const SIZE_WARNING_BYTES = 500 * 1024 * 1024; // 500MB

type Mode = 'single' | 'compare';
type PipelineMode = 'preset' | 'template' | 'custom';

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

interface Preset {
  steps_preset: string;
  label: string;
  description?: string;
}

interface PipelineStep {
  name: string;
  deps: string[];
  description: string;
  produces: string[];
  duration_estimate_s: number;
  config_schema?: Record<string, unknown>;
}

interface PipelineTemplate {
  name: string;
  description: string;
  steps: string[];
  preprocessing: string[];
}

interface PreprocessingOp {
  name: string;
  description: string;
}

interface ModelConfig {
  asr: {
    model_size: string;
    language: string;
  };
  diarization: {
    model_name: string;
  };
  device_preference: string[];
}

interface UserTemplate {
  name: string;
  steps: string[];
  preprocessing: string[];
  description?: string;
  created_at?: string;
  updated_at?: string;
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

  // Model configuration
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [selectedPreset, setSelectedPreset] = useState('full');
  const [modelConfig, setModelConfig] = useState<ModelConfig>({
    asr: { model_size: 'base', language: 'en' },
    diarization: { model_name: 'heuristic_diarization' },
    device_preference: ['mps', 'cpu']
  });
  
  // Per-step configuration (for custom mode)
  const [showStepConfig, setShowStepConfig] = useState(false);

  // Pipeline configuration
  const [pipelineMode, setPipelineMode] = useState<PipelineMode>('preset');
  const [pipelineSteps, setPipelineSteps] = useState<PipelineStep[]>([]);
  const [pipelineTemplates, setPipelineTemplates] = useState<PipelineTemplate[]>([]);
  const [preprocessingOps, setPreprocessingOps] = useState<PreprocessingOp[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState('');
  const [selectedSteps, setSelectedSteps] = useState<string[]>(['ingest', 'asr']);
  const [selectedPreprocessing, setSelectedPreprocessing] = useState<string[]>([]);
  const [resolvedSteps, setResolvedSteps] = useState<string[]>([]);
  
  // User-defined pipeline templates (server-side storage)
  const [userTemplates, setUserTemplates] = useState<UserTemplate[]>([]);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [newTemplateName, setNewTemplateName] = useState('');
  const [savingTemplate, setSavingTemplate] = useState(false);

  // Data
  const [useCases, setUseCases] = useState<UseCase[]>([]);
  const [candidates, setCandidates] = useState<Candidate[]>([]);

  // State
  const [loadingConfig, setLoadingConfig] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 1. Initial Load: Use Cases + Presets + Pipeline Config + User Templates
  useEffect(() => {
    (async () => {
      try {
        const [ucs, prsts, steps, templates, prepOps] = await Promise.all([
          api.getUseCases(),
          api.getPresets(),
          api.getPipelineSteps(),
          api.getPipelineTemplates(),
          api.getPipelinePreprocessing(),
        ]);
        setUseCases(ucs);
        setPresets(prsts);
        setPipelineSteps(steps);
        setPipelineTemplates(templates);
        setPreprocessingOps(prepOps);
        if (ucs.length > 0) setUseCaseId(ucs[0].use_case_id);
        if (prsts.length > 0) setSelectedPreset(prsts.find(p => p.steps_preset === 'full')?.steps_preset || prsts[0].steps_preset);
        if (templates.length > 0) setSelectedTemplate(templates[0].name);
        
        // Load user templates from server
        try {
          const userTpls = await api.getUserTemplates();
          setUserTemplates(userTpls);
        } catch {
          console.warn('Failed to load user templates from server');
        }
      } catch (e) {
        setError("Failed to load configuration");
      } finally {
        setLoadingConfig(false);
      }
    })();
  }, []);

  // Save user template (server-side)
  const saveUserTemplate = async () => {
    if (!newTemplateName.trim() || savingTemplate) return;
    
    setSavingTemplate(true);
    try {
      await api.saveUserTemplate({
        name: newTemplateName.trim(),
        steps: selectedSteps,
        preprocessing: selectedPreprocessing,
      });
      
      // Refresh templates from server
      const userTpls = await api.getUserTemplates();
      setUserTemplates(userTpls);
      setShowSaveDialog(false);
      setNewTemplateName('');
    } catch (e) {
      console.error('Failed to save template', e);
    } finally {
      setSavingTemplate(false);
    }
  };

  // Delete user template (server-side)
  const deleteUserTemplate = async (name: string) => {
    try {
      await api.deleteUserTemplate(name);
      setUserTemplates(userTemplates.filter(t => t.name !== name));
    } catch (e) {
      console.error('Failed to delete template', e);
    }
  };

  // Load user template
  const loadUserTemplate = (template: UserTemplate) => {
    setSelectedSteps(template.steps);
    setSelectedPreprocessing(template.preprocessing);
    setPipelineMode('custom');
  };

  // Resolve dependencies when custom steps change
  useEffect(() => {
    if (pipelineMode !== 'custom' || selectedSteps.length === 0) {
      setResolvedSteps([]);
      return;
    }
    (async () => {
      try {
        const result = await api.resolvePipelineSteps(selectedSteps);
        setResolvedSteps(result.resolved_steps);
      } catch (e) {
        console.error("Failed to resolve steps", e);
      }
    })();
  }, [pipelineMode, selectedSteps]);

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

  // Calculate estimated duration based on selected steps
  const estimatedDuration = useMemo(() => {
    const stepsToUse = pipelineMode === 'custom' && resolvedSteps.length > 0 
      ? resolvedSteps 
      : selectedSteps;
    
    const totalSeconds = stepsToUse.reduce((acc, stepName) => {
      const step = pipelineSteps.find(s => s.name === stepName);
      return acc + (step?.duration_estimate_s || 5);
    }, 0);
    
    // Return formatted string (per minute of audio)
    if (totalSeconds < 60) {
      return `~${totalSeconds}s per min`;
    }
    return `~${Math.round(totalSeconds / 60)}m per min`;
  }, [pipelineMode, resolvedSteps, selectedSteps, pipelineSteps]);

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

      // Build config overrides with per-step configuration
      const configOverrides = showAdvanced ? {
        asr: modelConfig.asr,
        diarization: modelConfig.diarization,
        device_preference: modelConfig.device_preference,
      } : undefined;

      // For simple single runs with custom pipeline, use direct workbench API
      if (mode === 'single' && showAdvanced && pipelineMode !== 'preset') {
        const result = await api.createWorkbenchRun(file, useCaseId, {
          stepsPreset: selectedPreset,
          steps: pipelineMode === 'custom' ? selectedSteps : undefined,
          preprocessing: selectedPreprocessing.length > 0 ? selectedPreprocessing : undefined,
          pipelineTemplate: pipelineMode === 'template' ? selectedTemplate : undefined,
          config: configOverrides,
        });
        navigate(`/runs/${result.run_id}`);
        return;
      }

      // Build pipeline options for experiment (if using custom pipeline in compare mode)
      const pipelineOptions = showAdvanced && pipelineMode !== 'preset' ? {
        steps: pipelineMode === 'custom' ? selectedSteps : undefined,
        preprocessing: selectedPreprocessing.length > 0 ? selectedPreprocessing : undefined,
        pipelineTemplate: pipelineMode === 'template' ? selectedTemplate : undefined,
      } : undefined;

      // For experiment-based runs (compare mode or preset mode)
      const exp = await api.createExperiment(
        file,
        useCaseId,
        candsToRun,
        {
          ...configOverrides,
          steps_preset: selectedPreset,
        },
        pipelineOptions
      );
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

        {/* Row 4: Advanced Configuration */}
        <div className="border-t pt-4">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-sm text-blue-600 hover:text-blue-800 font-medium"
          >
            {showAdvanced ? '▼ Hide Advanced Options' : '▶ Show Advanced Options'}
          </button>

          {showAdvanced && (
            <div className="mt-4 space-y-4 p-4 bg-gray-50 rounded-lg">
              {/* Pipeline Mode Selection */}
              <div>
                <span className="block font-semibold mb-2 text-sm">Pipeline Configuration</span>
                <div className="flex rounded bg-gray-200 p-1 mb-3">
                  <button
                    className={`flex-1 py-1 px-2 rounded text-xs font-medium transition-colors ${pipelineMode === 'preset' ? 'bg-white shadow text-gray-900' : 'text-gray-500 hover:text-gray-900'}`}
                    onClick={() => setPipelineMode('preset')}
                    disabled={isSubmitting}
                  >
                    Preset
                  </button>
                  <button
                    className={`flex-1 py-1 px-2 rounded text-xs font-medium transition-colors ${pipelineMode === 'template' ? 'bg-white shadow text-gray-900' : 'text-gray-500 hover:text-gray-900'}`}
                    onClick={() => setPipelineMode('template')}
                    disabled={isSubmitting}
                  >
                    Template
                  </button>
                  <button
                    className={`flex-1 py-1 px-2 rounded text-xs font-medium transition-colors ${pipelineMode === 'custom' ? 'bg-white shadow text-blue-600' : 'text-gray-500 hover:text-gray-900'}`}
                    onClick={() => setPipelineMode('custom')}
                    disabled={isSubmitting}
                  >
                    Custom Steps
                  </button>
                </div>

                {/* Preset Mode */}
                {pipelineMode === 'preset' && (
                  <select
                    value={selectedPreset}
                    onChange={(e) => setSelectedPreset(e.target.value)}
                    disabled={isSubmitting}
                    className="w-full border p-2 rounded text-sm"
                  >
                    {presets.map(p => (
                      <option key={p.steps_preset} value={p.steps_preset}>
                        {p.label} - {p.description}
                      </option>
                    ))}
                  </select>
                )}

                {/* Template Mode */}
                {pipelineMode === 'template' && (
                  <select
                    value={selectedTemplate}
                    onChange={(e) => setSelectedTemplate(e.target.value)}
                    disabled={isSubmitting}
                    className="w-full border p-2 rounded text-sm"
                  >
                    {pipelineTemplates.map(t => (
                      <option key={t.name} value={t.name}>
                        {t.name} - {t.description}
                      </option>
                    ))}
                  </select>
                )}

                {/* Custom Mode */}
                {pipelineMode === 'custom' && (
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      {pipelineSteps.filter(s => s.name !== 'bundle').map(step => (
                        <label
                          key={step.name}
                          title={`${step.description} (~${step.duration_estimate_s}s/min)`}
                          className={`flex items-center gap-2 p-2 rounded border cursor-pointer text-xs ${
                            selectedSteps.includes(step.name)
                              ? 'bg-blue-50 border-blue-300'
                              : 'bg-white border-gray-200 hover:border-gray-300'
                          }`}
                        >
                          <input
                            type="checkbox"
                            checked={selectedSteps.includes(step.name)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedSteps([...selectedSteps, step.name]);
                              } else {
                                setSelectedSteps(selectedSteps.filter(s => s !== step.name));
                              }
                            }}
                            disabled={isSubmitting || step.name === 'ingest'}
                            className="rounded"
                          />
                          <div className="flex flex-col">
                            <span className="font-medium">{step.name}</span>
                            <span className="text-gray-400 text-[10px]">{step.duration_estimate_s}s/min</span>
                          </div>
                        </label>
                      ))}
                    </div>
                    {resolvedSteps.length > 0 && (
                      <div className="text-xs text-gray-600 bg-white p-2 rounded border flex justify-between items-center">
                        <div>
                          <span className="font-medium">Execution order:</span>{' '}
                          {resolvedSteps.join(' → ')}
                        </div>
                        <span className="text-blue-600 font-medium">{estimatedDuration}</span>
                      </div>
                    )}
                    
                    {/* User Templates */}
                    <div className="flex flex-wrap items-center gap-2 pt-2 border-t">
                      <span className="text-xs text-gray-500">My Pipelines:</span>
                      {userTemplates.map(t => (
                        <div key={t.name} className="flex items-center gap-1 bg-purple-50 border border-purple-200 rounded px-2 py-1 text-xs">
                          <button
                            onClick={() => loadUserTemplate(t)}
                            className="font-medium text-purple-700 hover:text-purple-900"
                            title={`Steps: ${t.steps.join(', ')}`}
                          >
                            {t.name}
                          </button>
                          <button
                            onClick={() => deleteUserTemplate(t.name)}
                            className="text-purple-400 hover:text-red-500 ml-1"
                            title="Delete"
                          >
                            ×
                          </button>
                        </div>
                      ))}
                      {showSaveDialog ? (
                        <div className="flex items-center gap-1">
                          <input
                            type="text"
                            value={newTemplateName}
                            onChange={(e) => setNewTemplateName(e.target.value)}
                            placeholder="Pipeline name..."
                            className="border rounded px-2 py-1 text-xs w-32"
                            autoFocus
                            onKeyDown={(e) => e.key === 'Enter' && saveUserTemplate()}
                          />
                          <button onClick={saveUserTemplate} className="text-green-600 hover:text-green-800 text-xs font-medium">Save</button>
                          <button onClick={() => setShowSaveDialog(false)} className="text-gray-400 hover:text-gray-600 text-xs">Cancel</button>
                        </div>
                      ) : (
                        <button
                          onClick={() => setShowSaveDialog(true)}
                          className="text-xs text-blue-600 hover:text-blue-800 font-medium"
                          disabled={selectedSteps.length < 2}
                        >
                          + Save Current
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Preprocessing Operators (for template/custom modes) */}
              {pipelineMode !== 'preset' && (
                <div>
                  <span className="block font-semibold mb-2 text-sm">Preprocessing</span>
                  <div className="flex flex-wrap gap-2">
                    {preprocessingOps.map(op => (
                      <label
                        key={op.name}
                        className={`flex items-center gap-1 px-2 py-1 rounded border cursor-pointer text-xs ${
                          selectedPreprocessing.includes(op.name)
                            ? 'bg-green-50 border-green-300'
                            : 'bg-white border-gray-200 hover:border-gray-300'
                        }`}
                        title={op.description}
                      >
                        <input
                          type="checkbox"
                          checked={selectedPreprocessing.includes(op.name)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedPreprocessing([...selectedPreprocessing, op.name]);
                            } else {
                              setSelectedPreprocessing(selectedPreprocessing.filter(o => o !== op.name));
                            }
                          }}
                          disabled={isSubmitting}
                          className="rounded"
                        />
                        <span>{op.name.replace(/_/g, ' ')}</span>
                      </label>
                    ))}
                  </div>
                </div>
              )}

              {/* Model Configuration */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-sm">Model Configuration</span>
                  {pipelineMode === 'custom' && (
                    <button
                      type="button"
                      onClick={() => setShowStepConfig(!showStepConfig)}
                      className="text-xs text-blue-600 hover:text-blue-800"
                    >
                      {showStepConfig ? '▼ Simple View' : '▶ Per-Step Config'}
                    </button>
                  )}
                </div>
                
                {/* Simple Configuration (default) */}
                {!showStepConfig && (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <label className="block">
                      <span className="block font-semibold mb-1 text-sm">Model Size</span>
                      <select
                        value={modelConfig.asr.model_size}
                        onChange={(e) => setModelConfig({
                          ...modelConfig,
                          asr: { ...modelConfig.asr, model_size: e.target.value }
                        })}
                        disabled={isSubmitting}
                        className="w-full border p-2 rounded text-sm"
                      >
                        <option value="tiny">Tiny (fastest)</option>
                        <option value="base">Base</option>
                        <option value="small">Small</option>
                        <option value="medium">Medium</option>
                        <option value="large-v3">Large-v3 (most accurate)</option>
                      </select>
                    </label>

                    <label className="block">
                      <span className="block font-semibold mb-1 text-sm">Language</span>
                      <select
                        value={modelConfig.asr.language}
                        onChange={(e) => setModelConfig({
                          ...modelConfig,
                          asr: { ...modelConfig.asr, language: e.target.value }
                        })}
                        disabled={isSubmitting}
                        className="w-full border p-2 rounded text-sm"
                      >
                        <option value="en">English</option>
                        <option value="auto">Auto-detect</option>
                        <option value="es">Spanish</option>
                        <option value="fr">French</option>
                        <option value="de">German</option>
                        <option value="zh">Chinese</option>
                        <option value="ja">Japanese</option>
                        <option value="ko">Korean</option>
                        <option value="hi">Hindi</option>
                        <option value="pt">Portuguese</option>
                      </select>
                    </label>

                    <label className="block">
                      <span className="block font-semibold mb-1 text-sm">Device</span>
                      <select
                        value={modelConfig.device_preference[0]}
                        onChange={(e) => setModelConfig({
                          ...modelConfig,
                          device_preference: [e.target.value, 'cpu']
                        })}
                        disabled={isSubmitting}
                        className="w-full border p-2 rounded text-sm"
                      >
                        <option value="mps">Apple Silicon (MPS)</option>
                        <option value="cuda">NVIDIA GPU (CUDA)</option>
                        <option value="cpu">CPU</option>
                      </select>
                    </label>
                  </div>
                )}
                
                {/* Per-Step Configuration (advanced) */}
                {showStepConfig && pipelineMode === 'custom' && (
                  <div className="space-y-3 border rounded p-3 bg-white">
                    {/* ASR Configuration */}
                    {selectedSteps.includes('asr') && (
                      <div className="border-b pb-3">
                        <span className="block font-medium text-xs text-gray-600 mb-2">ASR (Speech-to-Text)</span>
                        <div className="grid grid-cols-2 gap-3">
                          <label className="block">
                            <span className="block text-xs text-gray-500 mb-1">Model Size</span>
                            <select
                              value={modelConfig.asr.model_size}
                              onChange={(e) => setModelConfig({
                                ...modelConfig,
                                asr: { ...modelConfig.asr, model_size: e.target.value }
                              })}
                              disabled={isSubmitting}
                              className="w-full border p-1.5 rounded text-xs"
                            >
                              <option value="tiny">Tiny (fastest)</option>
                              <option value="base">Base</option>
                              <option value="small">Small</option>
                              <option value="medium">Medium</option>
                              <option value="large-v3">Large-v3 (accurate)</option>
                            </select>
                          </label>
                          <label className="block">
                            <span className="block text-xs text-gray-500 mb-1">Language</span>
                            <select
                              value={modelConfig.asr.language}
                              onChange={(e) => setModelConfig({
                                ...modelConfig,
                                asr: { ...modelConfig.asr, language: e.target.value }
                              })}
                              disabled={isSubmitting}
                              className="w-full border p-1.5 rounded text-xs"
                            >
                              <option value="en">English</option>
                              <option value="auto">Auto-detect</option>
                              <option value="es">Spanish</option>
                              <option value="fr">French</option>
                              <option value="de">German</option>
                              <option value="zh">Chinese</option>
                              <option value="ja">Japanese</option>
                              <option value="ko">Korean</option>
                            </select>
                          </label>
                        </div>
                      </div>
                    )}
                    
                    {/* Diarization Configuration */}
                    {selectedSteps.includes('diarization') && (
                      <div className="border-b pb-3">
                        <span className="block font-medium text-xs text-gray-600 mb-2">Diarization (Speaker ID)</span>
                        <label className="block">
                          <span className="block text-xs text-gray-500 mb-1">Model</span>
                          <select
                            value={modelConfig.diarization.model_name}
                            onChange={(e) => setModelConfig({
                              ...modelConfig,
                              diarization: { model_name: e.target.value }
                            })}
                            disabled={isSubmitting}
                            className="w-full border p-1.5 rounded text-xs"
                          >
                            <option value="heuristic_diarization">Heuristic (fast)</option>
                            <option value="pyannote_diarization">Pyannote (accurate)</option>
                          </select>
                        </label>
                      </div>
                    )}
                    
                    {/* Device Configuration */}
                    <div>
                      <span className="block font-medium text-xs text-gray-600 mb-2">Compute Device</span>
                      <select
                        value={modelConfig.device_preference[0]}
                        onChange={(e) => setModelConfig({
                          ...modelConfig,
                          device_preference: [e.target.value, 'cpu']
                        })}
                        disabled={isSubmitting}
                        className="w-full border p-1.5 rounded text-xs"
                      >
                        <option value="mps">Apple Silicon (MPS)</option>
                        <option value="cuda">NVIDIA GPU (CUDA)</option>
                        <option value="cpu">CPU only</option>
                      </select>
                    </div>
                  </div>
                )}
              </div>

              <div className="text-xs text-gray-500 mt-2">
                These settings override the candidate defaults. Model size affects accuracy vs speed tradeoff.
              </div>
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
