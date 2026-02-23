import { describe, it, expect } from 'vitest';
import { deriveRunDetailViewModel } from './viewModel';
import type { ApiRunStatus } from './viewModel';

describe('deriveRunDetailViewModel', () => {
  // Helper to create minimal valid run status
  const createRun = (overrides: Partial<ApiRunStatus> = {}): ApiRunStatus => ({
    status: 'QUEUED',
    steps_completed: [],
    ...overrides,
  });

  describe('overall status mapping', () => {
    it('should map FAILED status to FAILED', () => {
      const run = createRun({ status: 'FAILED' });
      const vm = deriveRunDetailViewModel(run);
      expect(vm.overallStatus).toBe('FAILED');
    });

    it('should map CANCELLED status to FAILED', () => {
      const run = createRun({ status: 'CANCELLED' });
      const vm = deriveRunDetailViewModel(run);
      expect(vm.overallStatus).toBe('FAILED');
    });

    it('should map COMPLETED status to COMPLETED', () => {
      const run = createRun({ status: 'COMPLETED' });
      const vm = deriveRunDetailViewModel(run);
      expect(vm.overallStatus).toBe('COMPLETED');
    });

    it('should map RUNNING status to RUNNING', () => {
      const run = createRun({ status: 'RUNNING' });
      const vm = deriveRunDetailViewModel(run);
      expect(vm.overallStatus).toBe('RUNNING');
    });

    it('should map STALE status to RUNNING with isStale=true', () => {
      const run = createRun({ status: 'STALE' });
      const vm = deriveRunDetailViewModel(run);
      expect(vm.overallStatus).toBe('RUNNING');
      expect(vm.isStale).toBe(true);
    });

    it('should map QUEUED status to QUEUED', () => {
      const run = createRun({ status: 'QUEUED' });
      const vm = deriveRunDetailViewModel(run);
      expect(vm.overallStatus).toBe('QUEUED');
    });

    it('should force FAILED when failure_step is present regardless of status', () => {
      const run = createRun({
        status: 'RUNNING',
        failure_step: 'asr',
        error_message: 'ASR crashed',
      });
      const vm = deriveRunDetailViewModel(run);
      expect(vm.overallStatus).toBe('FAILED');
    });

    it('should force FAILED when error_message is present regardless of status', () => {
      const run = createRun({
        status: 'COMPLETED',
        error_message: 'Something went wrong',
      });
      const vm = deriveRunDetailViewModel(run);
      expect(vm.overallStatus).toBe('FAILED');
    });
  });

  describe('staleness detection', () => {
    it('should set isStale when is_stalled flag is true', () => {
      const run = createRun({ status: 'RUNNING', is_stalled: true });
      const vm = deriveRunDetailViewModel(run);
      expect(vm.isStale).toBe(true);
    });

    it('should set isStale when status is STALE', () => {
      const run = createRun({ status: 'STALE' });
      const vm = deriveRunDetailViewModel(run);
      expect(vm.isStale).toBe(true);
    });

    it('should not be stale for fresh runs', () => {
      const run = createRun({ status: 'RUNNING' });
      const vm = deriveRunDetailViewModel(run);
      expect(vm.isStale).toBe(false);
    });
  });

  describe('failure reasons', () => {
    it('should show error_message as primary reason when failed', () => {
      const run = createRun({
        status: 'FAILED',
        error_message: 'Out of memory',
      });
      const vm = deriveRunDetailViewModel(run);
      expect(vm.primaryReason).toBe('Out of memory');
    });

    it('should show failure_step as primary reason when no error_message', () => {
      const run = createRun({
        status: 'FAILED',
        failure_step: 'diarization',
      });
      const vm = deriveRunDetailViewModel(run);
      expect(vm.primaryReason).toBe('Failed at diarization');
    });

    it('should show generic message when no details available', () => {
      const run = createRun({ status: 'FAILED' });
      const vm = deriveRunDetailViewModel(run);
      expect(vm.primaryReason).toBe('Run failed (unknown error)');
    });

    it('should add staleness as secondary reason when failed and stale', () => {
      const run = createRun({
        status: 'FAILED',
        error_message: 'Crashed',
        is_stalled: true,
      });
      const vm = deriveRunDetailViewModel(run);
      expect(vm.primaryReason).toBe('Crashed');
      expect(vm.secondaryReason).toBe(
        'Worker stopped responding (No heartbeat)',
      );
    });

    it('should show staleness as primary reason when just stale (not failed)', () => {
      const run = createRun({
        status: 'RUNNING',
        is_stalled: true,
      });
      const vm = deriveRunDetailViewModel(run);
      expect(vm.primaryReason).toBe('Worker stopped responding');
      expect(vm.secondaryReason).toBe('No heartbeat recently');
    });
  });

  describe('pipeline steps', () => {
    it('should include all 8 pipeline steps in order', () => {
      const run = createRun();
      const vm = deriveRunDetailViewModel(run);

      expect(vm.pipelineSteps).toHaveLength(8);
      expect(vm.pipelineSteps[0].key).toBe('ingest');
      expect(vm.pipelineSteps[1].key).toBe('asr');
      expect(vm.pipelineSteps[2].key).toBe('diarization');
      expect(vm.pipelineSteps[3].key).toBe('alignment');
      expect(vm.pipelineSteps[4].key).toBe('chapters');
      expect(vm.pipelineSteps[5].key).toBe('summarize_by_speaker');
      expect(vm.pipelineSteps[6].key).toBe('action_items_assignee');
      expect(vm.pipelineSteps[7].key).toBe('bundle');
    });

    it('should mark steps as COMPLETED when in steps_completed', () => {
      const run = createRun({
        status: 'RUNNING',
        steps_completed: ['ingest', 'asr'],
      });
      const vm = deriveRunDetailViewModel(run);

      expect(vm.pipelineSteps[0].status).toBe('COMPLETED'); // ingest
      expect(vm.pipelineSteps[1].status).toBe('COMPLETED'); // asr
    });

    it('should mark failure_step as FAILED', () => {
      const run = createRun({
        status: 'FAILED',
        failure_step: 'diarization',
        steps_completed: ['ingest', 'asr'],
      });
      const vm = deriveRunDetailViewModel(run);

      expect(vm.pipelineSteps[0].status).toBe('COMPLETED'); // ingest
      expect(vm.pipelineSteps[1].status).toBe('COMPLETED'); // asr
      expect(vm.pipelineSteps[2].status).toBe('FAILED'); // diarization
    });

    it('should mark steps after failure as NOT_RUN', () => {
      const run = createRun({
        status: 'FAILED',
        failure_step: 'diarization',
        steps_completed: ['ingest', 'asr'],
      });
      const vm = deriveRunDetailViewModel(run);

      expect(vm.pipelineSteps[3].status).toBe('NOT_RUN'); // alignment
      expect(vm.pipelineSteps[4].status).toBe('NOT_RUN'); // chapters
      expect(vm.pipelineSteps[7].status).toBe('NOT_RUN'); // bundle
    });

    it('should mark current_step as RUNNING', () => {
      const run = createRun({
        status: 'RUNNING',
        current_step: 'diarization',
        steps_completed: ['ingest', 'asr'],
      });
      const vm = deriveRunDetailViewModel(run);

      expect(vm.pipelineSteps[0].status).toBe('COMPLETED'); // ingest
      expect(vm.pipelineSteps[1].status).toBe('COMPLETED'); // asr
      expect(vm.pipelineSteps[2].status).toBe('RUNNING'); // diarization
    });

    it('should mark steps after current as PENDING', () => {
      const run = createRun({
        status: 'RUNNING',
        current_step: 'diarization',
        steps_completed: ['ingest', 'asr'],
      });
      const vm = deriveRunDetailViewModel(run);

      expect(vm.pipelineSteps[3].status).toBe('PENDING'); // alignment
      expect(vm.pipelineSteps[7].status).toBe('PENDING'); // bundle
    });

    it('should detect completed steps by artifacts when current_step is provided', () => {
      // Artifacts only mark steps as COMPLETED when there's a validated current_step
      // and the step comes before it in the pipeline
      const run = createRun({
        status: 'RUNNING',
        current_step: 'diarization',
        steps: [
          {
            name: 'ingest',
            status: 'COMPLETED',
            artifacts: [
              {
                id: '1',
                filename: 'audio.wav',
                role: 'audio',
                produced_by: 'ingest',
                size_bytes: 1000,
                downloadable: true,
              },
            ],
          },
          {
            name: 'asr',
            status: 'COMPLETED',
            artifacts: [
              {
                id: '2',
                filename: 'transcript.txt',
                role: 'transcript',
                produced_by: 'asr',
                size_bytes: 500,
                downloadable: true,
              },
            ],
          },
        ],
      });
      const vm = deriveRunDetailViewModel(run);

      // Steps before current_step with artifacts are COMPLETED
      expect(vm.pipelineSteps[0].status).toBe('COMPLETED'); // ingest
      expect(vm.pipelineSteps[1].status).toBe('COMPLETED'); // asr
      expect(vm.pipelineSteps[2].status).toBe('RUNNING'); // diarization (current)
    });

    it('should handle case-insensitive current_step normalization', () => {
      const run = createRun({
        status: 'RUNNING',
        current_step: 'ASR', // uppercase
        steps_completed: ['ingest'],
      });
      const vm = deriveRunDetailViewModel(run);

      expect(vm.pipelineSteps[1].status).toBe('RUNNING');
    });

    it('should handle transcription -> asr alias', () => {
      const run = createRun({
        status: 'RUNNING',
        current_step: 'transcription',
        steps_completed: ['ingest'],
      });
      const vm = deriveRunDetailViewModel(run);

      expect(vm.pipelineSteps[1].status).toBe('RUNNING');
    });

    it('should surface unknown current_step as secondary reason', () => {
      const run = createRun({
        status: 'RUNNING',
        current_step: 'unknown_step_xyz',
      });
      const vm = deriveRunDetailViewModel(run);

      expect(vm.secondaryReason).toBe('Unknown current_step: unknown_step_xyz');
    });

    it('should infer RUNNING when no current_step but previous steps completed', () => {
      const run = createRun({
        status: 'RUNNING',
        steps_completed: ['ingest'],
      });
      const vm = deriveRunDetailViewModel(run);

      // First non-completed step should be RUNNING
      expect(vm.pipelineSteps[1].status).toBe('RUNNING');
    });

    it('should mark all steps COMPLETED when run is complete', () => {
      const run = createRun({
        status: 'COMPLETED',
        steps_completed: [
          'ingest',
          'asr',
          'diarization',
          'alignment',
          'chapters',
          'summarize_by_speaker',
          'action_items_assignee',
          'bundle',
        ],
      });
      const vm = deriveRunDetailViewModel(run);

      vm.pipelineSteps.forEach((step) => {
        expect(step.status).toBe('COMPLETED');
      });
    });
  });

  describe('step labels', () => {
    it('should have human-readable labels for all steps', () => {
      const run = createRun();
      const vm = deriveRunDetailViewModel(run);

      expect(vm.pipelineSteps[0].label).toBe('Ingest audio');
      expect(vm.pipelineSteps[1].label).toBe('Speech recognition');
      expect(vm.pipelineSteps[2].label).toBe('Speaker identification');
      expect(vm.pipelineSteps[7].label).toBe('Bundle results');
    });

    it('should fallback to key when label is unknown', () => {
      // This shouldn't happen with current implementation but good to verify
      const run = createRun({
        steps: [{ name: 'unknown_step', status: 'PENDING', artifacts: [] }],
      });
      const vm = deriveRunDetailViewModel(run);

      // All steps in ORDERED_PIPELINE_KEYS have labels defined
      vm.pipelineSteps.forEach((step) => {
        expect(step.label).toBeDefined();
        expect(step.label.length).toBeGreaterThan(0);
      });
    });
  });

  describe('resolved device', () => {
    it('should extract device from ASR step resolved_config', () => {
      const run = createRun({
        status: 'COMPLETED',
        steps: [
          {
            name: 'asr',
            status: 'COMPLETED',
            artifacts: [],
            resolved_config: {
              model_id: 'whisper',
              source: 'openai',
              device: 'cuda',
              language: 'en',
            },
          },
        ],
      });
      const vm = deriveRunDetailViewModel(run);

      expect(vm.resolvedDevice).toBe('cuda');
    });

    it('should handle mps device', () => {
      const run = createRun({
        status: 'COMPLETED',
        steps: [
          {
            name: 'asr',
            status: 'COMPLETED',
            artifacts: [],
            resolved_config: {
              model_id: 'whisper',
              source: 'openai',
              device: 'mps',
              language: 'en',
            },
          },
        ],
      });
      const vm = deriveRunDetailViewModel(run);

      expect(vm.resolvedDevice).toBe('mps');
    });

    it('should be undefined when ASR has no resolved_config', () => {
      const run = createRun({
        status: 'COMPLETED',
        steps: [{ name: 'asr', status: 'COMPLETED', artifacts: [] }],
      });
      const vm = deriveRunDetailViewModel(run);

      expect(vm.resolvedDevice).toBeUndefined();
    });

    it('should be undefined when no ASR step', () => {
      const run = createRun({ status: 'COMPLETED' });
      const vm = deriveRunDetailViewModel(run);

      expect(vm.resolvedDevice).toBeUndefined();
    });
  });

  describe('edge cases', () => {
    it('should handle empty steps_completed array', () => {
      const run = createRun({ status: 'QUEUED' });
      const vm = deriveRunDetailViewModel(run);

      expect(
        vm.pipelineSteps.every(
          (s) => s.status === 'PENDING' || s.status === 'UNKNOWN',
        ),
      ).toBe(true);
    });

    it('should handle null current_step', () => {
      const run = createRun({
        status: 'RUNNING',
        current_step: null as any,
      });
      const vm = deriveRunDetailViewModel(run);

      expect(vm.overallStatus).toBe('RUNNING');
    });

    it('should handle missing steps array', () => {
      const run = createRun({ status: 'RUNNING' });
      delete (run as any).steps;
      const vm = deriveRunDetailViewModel(run);

      expect(vm.pipelineSteps).toHaveLength(8);
    });

    it('should handle failure at first step', () => {
      const run = createRun({
        status: 'FAILED',
        failure_step: 'ingest',
      });
      const vm = deriveRunDetailViewModel(run);

      expect(vm.pipelineSteps[0].status).toBe('FAILED');
      expect(vm.pipelineSteps[1].status).toBe('NOT_RUN');
      expect(vm.pipelineSteps[7].status).toBe('NOT_RUN');
    });

    it('should handle failure at last step', () => {
      const run = createRun({
        status: 'FAILED',
        failure_step: 'bundle',
        steps_completed: [
          'ingest',
          'asr',
          'diarization',
          'alignment',
          'chapters',
          'summarize_by_speaker',
          'action_items_assignee',
        ],
      });
      const vm = deriveRunDetailViewModel(run);

      expect(vm.pipelineSteps[6].status).toBe('COMPLETED');
      expect(vm.pipelineSteps[7].status).toBe('FAILED');
    });
  });
});
