import { describe, it, expect, vi, beforeEach } from 'vitest';
import axios from 'axios';
import { api } from './api';

// Mock axios
vi.mock('axios');
const mockedAxios = vi.mocked(axios, true);

describe('API Client', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('getRuns', () => {
    it('should fetch runs without refresh param by default', async () => {
      const mockRuns = [
        { run_id: 'run-1', status: 'COMPLETED', input_filename: 'test.wav' }
      ];
      mockedAxios.get.mockResolvedValueOnce({ data: mockRuns });

      const result = await api.getRuns();

      expect(mockedAxios.get).toHaveBeenCalledWith('/api/runs', { params: { refresh: false } });
      expect(result).toEqual(mockRuns);
    });

    it('should fetch runs with refresh=true when specified', async () => {
      const mockRuns = [{ run_id: 'run-1', status: 'COMPLETED', input_filename: 'test.wav' }];
      mockedAxios.get.mockResolvedValueOnce({ data: mockRuns });

      await api.getRuns(true);

      expect(mockedAxios.get).toHaveBeenCalledWith('/api/runs', { params: { refresh: true } });
    });
  });

  describe('refreshRuns', () => {
    it('should POST to refresh endpoint', async () => {
      const mockRuns = [{ run_id: 'run-1', status: 'COMPLETED', input_filename: 'test.wav' }];
      mockedAxios.post.mockResolvedValueOnce({ data: mockRuns });

      const result = await api.refreshRuns();

      expect(mockedAxios.post).toHaveBeenCalledWith('/api/runs/refresh');
      expect(result).toEqual(mockRuns);
    });
  });

  describe('searchRun', () => {
    it('should search with query and default limit', async () => {
      const mockResults = {
        query: 'hello',
        results: [{ segment_id: '1', text: 'hello world', match_start: 0, match_end: 5, start_s: 0, end_s: 2 }]
      };
      mockedAxios.get.mockResolvedValueOnce({ data: mockResults });

      const result = await api.searchRun('run-1', 'hello');

      expect(mockedAxios.get).toHaveBeenCalledWith('/api/runs/run-1/search', {
        params: { q: 'hello', limit: 50 },
        signal: undefined
      });
      expect(result).toEqual(mockResults);
    });

    it('should accept custom limit and abort signal', async () => {
      const controller = new AbortController();
      mockedAxios.get.mockResolvedValueOnce({ data: { query: 'test', results: [] } });

      await api.searchRun('run-1', 'test', 10, controller.signal);

      expect(mockedAxios.get).toHaveBeenCalledWith('/api/runs/run-1/search', {
        params: { q: 'test', limit: 10 },
        signal: controller.signal
      });
    });
  });

  describe('getRunDetail', () => {
    it('should fetch run details', async () => {
      const mockDetail = {
        summary: { run_id: 'run-1', status: 'COMPLETED', input_filename: 'test.wav', steps_completed: [] },
        manifest: { schema_version: '1.0' }
      };
      mockedAxios.get.mockResolvedValueOnce({ data: mockDetail });

      const result = await api.getRunDetail('run-1');

      expect(mockedAxios.get).toHaveBeenCalledWith('/api/runs/run-1');
      expect(result).toEqual(mockDetail);
    });
  });

  describe('getRunStatus', () => {
    it('should fetch run status', async () => {
      const mockStatus = {
        run_id: 'run-1',
        status: 'RUNNING',
        steps_completed: ['ingest'],
        current_step: 'asr'
      };
      mockedAxios.get.mockResolvedValueOnce({ data: mockStatus });

      const result = await api.getRunStatus('run-1');

      expect(mockedAxios.get).toHaveBeenCalledWith('/api/runs/run-1/status');
      expect(result).toEqual(mockStatus);
    });
  });

  describe('getTranscript', () => {
    it('should fetch transcript', async () => {
      const mockTranscript = {
        run_id: 'run-1',
        segments: [{ start_s: 0, end_s: 2, text: 'Hello world', speaker: 'A' }],
        chapters: [{ start_s: 0, end_s: 10, title: 'Introduction' }]
      };
      mockedAxios.get.mockResolvedValueOnce({ data: mockTranscript });

      const result = await api.getTranscript('run-1');

      expect(mockedAxios.get).toHaveBeenCalledWith('/api/runs/run-1/transcript');
      expect(result).toEqual(mockTranscript);
    });
  });

  describe('URL generators', () => {
    it('getAudioUrl should return correct URL', () => {
      expect(api.getAudioUrl('run-1')).toBe('/api/runs/run-1/audio');
    });

    it('getMeetingPackZipUrl should return correct URL', () => {
      expect(api.getMeetingPackZipUrl('run-1')).toBe('/api/runs/run-1/bundle.zip');
    });

    it('getSessionBundleZipUrl should return correct URL', () => {
      expect(api.getSessionBundleZipUrl('run-1')).toBe('/api/runs/run-1/session_bundle.zip');
    });

    it('getMeetingPackArtifactUrl should encode artifact name', () => {
      expect(api.getMeetingPackArtifactUrl('run-1', 'file with spaces.txt'))
        .toBe('/api/runs/run-1/bundle/file%20with%20spaces.txt');
    });

    it('getMeetingPackArtifactPreviewUrl should include maxBytes param', () => {
      expect(api.getMeetingPackArtifactPreviewUrl('run-1', 'transcript.txt', 1000))
        .toBe('/api/runs/run-1/meeting-pack/artifacts/transcript.txt/preview?max_bytes=1000');
    });

    it('getMeetingPackArtifactPreviewUrl should use default maxBytes', () => {
      expect(api.getMeetingPackArtifactPreviewUrl('run-1', 'transcript.txt'))
        .toBe('/api/runs/run-1/meeting-pack/artifacts/transcript.txt/preview?max_bytes=50000');
    });
  });

  describe('createExperiment', () => {
    it('should create experiment with file and use_case_id', async () => {
      const mockFile = new File(['audio'], 'test.wav', { type: 'audio/wav' });
      const mockResponse = { experiment_id: 'exp-1', status: 'created' };
      mockedAxios.post.mockResolvedValueOnce({ data: mockResponse });

      const result = await api.createExperiment(mockFile, 'meeting_smoke');

      expect(mockedAxios.post).toHaveBeenCalledWith(
        '/api/experiments',
        expect.any(FormData)
      );
      
      // Verify FormData contents
      const formData = mockedAxios.post.mock.calls[0][1] as FormData;
      expect(formData.get('file')).toBe(mockFile);
      expect(formData.get('use_case_id')).toBe('meeting_smoke');
      expect(result).toEqual(mockResponse);
    });

    it('should include candidate_ids when provided', async () => {
      const mockFile = new File(['audio'], 'test.wav', { type: 'audio/wav' });
      mockedAxios.post.mockResolvedValueOnce({ data: {} });

      await api.createExperiment(mockFile, 'meeting_smoke', ['A', 'B']);

      const formData = mockedAxios.post.mock.calls[0][1] as FormData;
      expect(formData.get('candidate_ids')).toBe('A,B');
    });

    it('should include config when provided', async () => {
      const mockFile = new File(['audio'], 'test.wav', { type: 'audio/wav' });
      const config = { model: 'whisper', language: 'en' };
      mockedAxios.post.mockResolvedValueOnce({ data: {} });

      await api.createExperiment(mockFile, 'meeting_smoke', undefined, config);

      const formData = mockedAxios.post.mock.calls[0][1] as FormData;
      expect(formData.get('config')).toBe(JSON.stringify(config));
    });
  });

  describe('experiment operations', () => {
    it('getExperiment should fetch experiment details', async () => {
      const mockExp = { experiment_id: 'exp-1', status: 'active' };
      mockedAxios.get.mockResolvedValueOnce({ data: mockExp });

      const result = await api.getExperiment('exp-1');

      expect(mockedAxios.get).toHaveBeenCalledWith('/api/experiments/exp-1');
      expect(result).toEqual(mockExp);
    });

    it('startExperimentAll should start all runs', async () => {
      mockedAxios.post.mockResolvedValueOnce({ data: { started: 2 } });

      await api.startExperimentAll('exp-1');

      expect(mockedAxios.post).toHaveBeenCalledWith('/api/experiments/exp-1/runs/start-all');
    });

    it('startExperimentNext should start next run', async () => {
      mockedAxios.post.mockResolvedValueOnce({ data: { run_id: 'run-2' } });

      await api.startExperimentNext('exp-1');

      expect(mockedAxios.post).toHaveBeenCalledWith('/api/experiments/exp-1/runs/start');
    });
  });

  describe('comparison', () => {
    it('getExperimentComparison should fetch artifact comparison', async () => {
      const mockComparison = { left: 'content A', right: 'content B' };
      mockedAxios.get.mockResolvedValueOnce({ data: mockComparison });

      const result = await api.getExperimentComparison('exp-1', 'run-A', 'run-B', 'transcript.txt');

      expect(mockedAxios.get).toHaveBeenCalledWith('/api/experiments/exp-1/compare', {
        params: { left: 'run-A', right: 'run-B', artifact: 'transcript.txt', max_bytes: 200000 }
      });
      expect(result).toEqual(mockComparison);
    });

    it('getExperimentComparisonResults should fetch comparison summary', async () => {
      const mockResults = {
        schema_version: 'v1',
        experiment_id: 'exp-1',
        candidates: { A: { label: 'A', run_id: 'run-A', status: 'COMPLETED' }, B: { label: 'B', run_id: 'run-B', status: 'COMPLETED' } },
        readiness: { comparable: true },
        verdicts: { overall: 'A_BETTER', reasons: ['Better accuracy'] }
      };
      mockedAxios.get.mockResolvedValueOnce({ data: mockResults });

      const result = await api.getExperimentComparisonResults('exp-1');

      expect(mockedAxios.get).toHaveBeenCalledWith('/api/experiments/exp-1/compare-results');
      expect(result).toEqual(mockResults);
    });
  });

  describe('workbench', () => {
    it('getPresets should fetch available presets', async () => {
      const mockPresets = [
        { steps_preset: 'full', label: 'Full Pipeline' },
        { steps_preset: 'fast', label: 'Fast ASR Only' }
      ];
      mockedAxios.get.mockResolvedValueOnce({ data: mockPresets });

      const result = await api.getPresets();

      expect(mockedAxios.get).toHaveBeenCalledWith('/api/workbench/presets');
      expect(result).toEqual(mockPresets);
    });

    it('getUseCases should fetch use cases', async () => {
      const mockUseCases = [
        { use_case_id: 'meeting', title: 'Meeting Analysis', description: 'Full meeting pipeline', supported_steps_presets: ['full', 'fast'] }
      ];
      mockedAxios.get.mockResolvedValueOnce({ data: mockUseCases });

      const result = await api.getUseCases();

      expect(mockedAxios.get).toHaveBeenCalledWith('/api/use-cases');
      expect(result).toEqual(mockUseCases);
    });

    it('getCandidatesForUseCase should fetch candidates', async () => {
      const mockCandidates = [
        { candidate_id: 'whisper-large', label: 'Whisper Large', use_case_id: 'meeting', steps_preset: 'full', params: {}, expected_artifacts: ['transcript'] }
      ];
      mockedAxios.get.mockResolvedValueOnce({ data: mockCandidates });

      const result = await api.getCandidatesForUseCase('meeting');

      expect(mockedAxios.get).toHaveBeenCalledWith('/api/use-cases/meeting/candidates');
      expect(result).toEqual(mockCandidates);
    });
  });

  describe('meeting pack', () => {
    it('getMeetingPackManifest should fetch manifest', async () => {
      const mockManifest = {
        schema_version: '1.0',
        run_id: 'run-1',
        generated_at: '2024-01-01T00:00:00Z',
        artifacts: [{ name: 'transcript.txt', rel_path: 'transcript.txt', bytes: 100, sha256: 'abc', content_type: 'text/plain' }],
        absent: []
      };
      mockedAxios.get.mockResolvedValueOnce({ data: mockManifest });

      const result = await api.getMeetingPackManifest('run-1');

      expect(mockedAxios.get).toHaveBeenCalledWith('/api/runs/run-1/bundle');
      expect(result).toEqual(mockManifest);
    });

    it('getRunResults should fetch results summary', async () => {
      const mockResults = {
        schema_version: 'v1',
        run_id: 'run-1',
        experiment_id: 'exp-1',
        candidate_label: 'A',
        status: 'COMPLETED',
        executed_steps: ['ingest', 'asr'],
        metrics: { duration_s: 10, word_count: 100 },
        quality_flags: { is_partial: false, is_empty: false, warnings: [] },
        provenance: { computed_at: '2024-01-01T00:00:00Z', semantics_version: 'v1' }
      };
      mockedAxios.get.mockResolvedValueOnce({ data: mockResults });

      const result = await api.getRunResults('run-1');

      expect(mockedAxios.get).toHaveBeenCalledWith('/api/runs/run-1/results');
      expect(result).toEqual(mockResults);
    });
  });

  describe('run control', () => {
    it('killRun should POST to kill endpoint', async () => {
      mockedAxios.post.mockResolvedValueOnce({ data: undefined });

      await api.killRun('run-1');

      expect(mockedAxios.post).toHaveBeenCalledWith('/api/runs/run-1/kill');
    });

    it('retryRun should POST to retry endpoint without from_step', async () => {
      mockedAxios.post.mockResolvedValueOnce({ data: { run_id: 'run-2', console_url: '/runs/run-2' } });

      const result = await api.retryRun('run-1');

      expect(mockedAxios.post).toHaveBeenCalledWith('/api/runs/run-1/retry', { from_step: undefined });
      expect(result.run_id).toBe('run-2');
    });

    it('retryRun should include from_step when provided', async () => {
      mockedAxios.post.mockResolvedValueOnce({ data: { run_id: 'run-2', console_url: '/runs/run-2' } });

      await api.retryRun('run-1', 'diarization');

      expect(mockedAxios.post).toHaveBeenCalledWith('/api/runs/run-1/retry', { from_step: 'diarization' });
    });
  });

  describe('pipeline configuration', () => {
    it('getPipelineSteps should fetch available steps', async () => {
      const mockSteps = [
        { name: 'ingest', deps: [], description: 'Audio ingestion', produces: ['audio'], duration_estimate_s: 5 }
      ];
      mockedAxios.get.mockResolvedValueOnce({ data: mockSteps });

      const result = await api.getPipelineSteps();

      expect(mockedAxios.get).toHaveBeenCalledWith('/api/pipelines/steps');
      expect(result).toEqual(mockSteps);
    });

    it('getPipelinePreprocessing should fetch preprocessing ops', async () => {
      const mockOps = [
        { name: 'trim_silence', description: 'Remove silence', params: { threshold: { type: 'number', default: -40 } } }
      ];
      mockedAxios.get.mockResolvedValueOnce({ data: mockOps });

      const result = await api.getPipelinePreprocessing();

      expect(mockedAxios.get).toHaveBeenCalledWith('/api/pipelines/preprocessing');
      expect(result).toEqual(mockOps);
    });

    it('getPipelineTemplates should fetch templates', async () => {
      const mockTemplates = [
        { name: 'full_meeting', description: 'Full pipeline', steps: ['ingest', 'asr', 'diarization'], preprocessing: ['trim_silence'] }
      ];
      mockedAxios.get.mockResolvedValueOnce({ data: mockTemplates });

      const result = await api.getPipelineTemplates();

      expect(mockedAxios.get).toHaveBeenCalledWith('/api/pipelines/templates');
      expect(result).toEqual(mockTemplates);
    });

    it('resolvePipelineSteps should POST steps to resolve dependencies', async () => {
      const mockResponse = {
        requested_steps: ['diarization'],
        resolved_steps: ['ingest', 'diarization'],
        added_dependencies: ['ingest']
      };
      mockedAxios.post.mockResolvedValueOnce({ data: mockResponse });

      const result = await api.resolvePipelineSteps(['diarization']);

      expect(mockedAxios.post).toHaveBeenCalledWith('/api/pipelines/resolve', { steps: ['diarization'] });
      expect(result).toEqual(mockResponse);
    });
  });

  describe('createWorkbenchRun', () => {
    it('should create run with minimal options', async () => {
      const mockFile = new File(['audio'], 'test.wav', { type: 'audio/wav' });
      const mockResponse = { run_id: 'run-1', run_dir: '/runs/run-1', console_url: '/runs/run-1' };
      mockedAxios.post.mockResolvedValueOnce({ data: mockResponse });

      const result = await api.createWorkbenchRun(mockFile, 'meeting_smoke');

      expect(mockedAxios.post).toHaveBeenCalledWith(
        '/api/workbench/runs',
        expect.any(FormData)
      );

      const formData = mockedAxios.post.mock.calls[0][1] as FormData;
      expect(formData.get('file')).toBe(mockFile);
      expect(formData.get('use_case_id')).toBe('meeting_smoke');
      expect(formData.get('steps_preset')).toBe('full');
      expect(result).toEqual(mockResponse);
    });

    it('should include all optional parameters when provided', async () => {
      const mockFile = new File(['audio'], 'test.wav', { type: 'audio/wav' });
      mockedAxios.post.mockResolvedValueOnce({ data: {} });

      await api.createWorkbenchRun(mockFile, 'meeting_smoke', {
        stepsPreset: 'fast',
        steps: ['ingest', 'asr'],
        preprocessing: ['trim_silence'],
        pipelineTemplate: 'quick_summary',
        config: { model: 'whisper' }
      });

      const formData = mockedAxios.post.mock.calls[0][1] as FormData;
      expect(formData.get('steps_preset')).toBe('fast');
      expect(formData.get('steps')).toBe('ingest,asr');
      expect(formData.get('preprocessing')).toBe('trim_silence');
      expect(formData.get('pipeline_template')).toBe('quick_summary');
      expect(formData.get('config')).toBe(JSON.stringify({ model: 'whisper' }));
    });

    it('should not include empty arrays', async () => {
      const mockFile = new File(['audio'], 'test.wav', { type: 'audio/wav' });
      mockedAxios.post.mockResolvedValueOnce({ data: {} });

      await api.createWorkbenchRun(mockFile, 'meeting_smoke', {
        steps: [],
        preprocessing: []
      });

      const formData = mockedAxios.post.mock.calls[0][1] as FormData;
      expect(formData.get('steps')).toBeNull();
      expect(formData.get('preprocessing')).toBeNull();
    });
  });
});
