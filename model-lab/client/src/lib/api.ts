import axios from 'axios';

const API_BASE = '/api';

export interface RunSummary {
  run_id: string;
  status: string;
  started_at?: string;
  input_filename: string;
  duration?: number;
  steps_completed: string[];
}

export interface Segment {
  start_s: number;
  end_s: number;
  text: string;
  speaker?: string;
}

export interface Chapter {
  start_s: number;
  end_s: number;
  title: string;
}

export interface RunDetail {
  run_id: string;
  segments: Segment[];
  chapters: Chapter[];
}

export interface SearchResult {
  segment_id: string;
  start_s: number;
  end_s: number;
  text: string;
  match_start: number;
  match_end: number;
}

export interface SearchResults {
  query: string;
  results: SearchResult[];
}

export interface MeetingPackArtifact {
  name: string;
  rel_path: string;
  // Back-compat if older bundles are present on disk.
  path?: string;
  bytes: number;
  sha256: string;
  content_type: string;
}

export interface MeetingPackManifest {
  schema_version: string;
  run_id: string;
  generated_at: string;
  artifacts: MeetingPackArtifact[];
  absent: { name: string; reason: string }[];
}

export const api = {
  getRuns: async (refresh = false): Promise<RunSummary[]> => {
    const res = await axios.get(`${API_BASE}/runs`, { params: { refresh } });
    return res.data;
  },

  refreshRuns: async (): Promise<RunSummary[]> => {
    const res = await axios.post(`${API_BASE}/runs/refresh`);
    return res.data;
  },

  searchRun: async (runId: string, query: string, limit: number = 50, signal?: AbortSignal): Promise<SearchResults> => {
    const res = await axios.get(`${API_BASE}/runs/${runId}/search`, {
      params: { q: query, limit },
      signal
    });
    return res.data;
  },
  
  getRunDetail: async (runId: string): Promise<{summary: RunSummary, manifest: any}> => {
      const res = await axios.get(`${API_BASE}/runs/${runId}`);
      return res.data;
  },

  getRunStatus: async (runId: string): Promise<{
    run_id: string,
    status: string,
    started_at?: string,
    steps_completed: string[],
    current_step?: string | null,
    updated_at?: string,
    error_code?: string,
    error_message?: string
  }> => {
    const res = await axios.get(`${API_BASE}/runs/${runId}/status`);
    return res.data;
  },

  getTranscript: async (runId: string): Promise<RunDetail> => {
    const res = await axios.get(`${API_BASE}/runs/${runId}/transcript`);
    return res.data;
  },

  getAudioUrl: (runId: string) => `${API_BASE}/runs/${runId}/audio`,
  getMeetingPackZipUrl: (runId: string) => `${API_BASE}/runs/${runId}/bundle.zip`,
  getSessionBundleZipUrl: (runId: string) => `${API_BASE}/runs/${runId}/session_bundle.zip`,
  getMeetingPackArtifactUrl: (runId: string, artifactName: string) => `${API_BASE}/runs/${runId}/bundle/${encodeURIComponent(artifactName)}`,
  getMeetingPackArtifactPreviewUrl: (runId: string, name: string, maxBytes = 50_000): string =>
    `${API_BASE}/runs/${runId}/meeting-pack/artifacts/${encodeURIComponent(name)}/preview?max_bytes=${maxBytes}`,

  // Experiments
  createExperiment: async (file: File, use_case_id: string, candidate_ids?: string[]): Promise<any> => {
    const form = new FormData();
    form.append('file', file);
    form.append('use_case_id', use_case_id);
    if (candidate_ids && candidate_ids.length >= 1) {
      form.append('candidate_ids', candidate_ids.join(','));
    }
    const res = await axios.post(`${API_BASE}/experiments`, form);
    return res.data;
  },

  getExperiment: async (experimentId: string): Promise<any> => {
    const res = await axios.get(`${API_BASE}/experiments/${experimentId}`);
    return res.data;
  },

  startExperimentAll: async (experimentId: string): Promise<any> => {
    const res = await axios.post(`${API_BASE}/experiments/${experimentId}/runs/start-all`);
    return res.data;
  },

  startExperimentNext: async (experimentId: string): Promise<any> => {
    const res = await axios.post(`${API_BASE}/experiments/${experimentId}/runs/start`);
    return res.data;
  },

  getExperimentComparison: async (experimentId: string, leftRunId: string, rightRunId: string, artifact: string, maxBytes = 200000): Promise<any> => {
    const res = await axios.get(`${API_BASE}/experiments/${experimentId}/compare`, {
        params: { left: leftRunId, right: rightRunId, artifact, max_bytes: maxBytes }
    });
    return res.data;
  },

  getPresets: async (): Promise<{ steps_preset: string; label: string; description?: string }[]> => {
    const res = await axios.get(`${API_BASE}/workbench/presets`);
    return res.data;
  },

  getUseCases: async (): Promise<{ use_case_id: string; title: string; description: string; supported_steps_presets: string[] }[]> => {
    const res = await axios.get(`${API_BASE}/use-cases`);
    return res.data;
  },

  getCandidatesForUseCase: async (useCaseId: string): Promise<{ candidate_id: string; label: string; use_case_id: string; steps_preset: string; params: any; expected_artifacts: string[]; description?: string }[]> => {
    const res = await axios.get(`${API_BASE}/use-cases/${useCaseId}/candidates`);
    return res.data;
  },

  getMeetingPackManifest: async (runId: string): Promise<MeetingPackManifest> => {
    const res = await axios.get(`${API_BASE}/runs/${runId}/bundle`);
    return res.data;
  },
};
