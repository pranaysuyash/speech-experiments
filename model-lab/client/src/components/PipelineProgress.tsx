import { CheckCircle2, Circle, Loader2, XCircle, SkipForward } from 'lucide-react';

export interface StepProgressData {
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  progressPct: number;
  message?: string;
  durationMs?: number;
  estimatedRemainingS?: number;
  startedAt?: string;
  endedAt?: string;
}

interface PipelineProgressProps {
  steps: StepProgressData[];
  className?: string;
}

const STEP_LABELS: Record<string, string> = {
  ingest: 'Ingest audio',
  asr: 'Speech recognition',
  diarization: 'Speaker identification',
  alignment: 'Text alignment',
  chapters: 'Chapters',
  summarize_by_speaker: 'Summary',
  action_items_assignee: 'Action items',
  bundle: 'Bundle results',
};

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${remainingSeconds}s`;
}

function formatEstimate(seconds: number): string {
  if (seconds < 60) return `~${seconds}s remaining`;
  const minutes = Math.floor(seconds / 60);
  return `~${minutes}m remaining`;
}

function StatusIcon({ status }: { status: StepProgressData['status'] }) {
  switch (status) {
    case 'completed':
      return <CheckCircle2 className="w-5 h-5 text-green-500" />;
    case 'running':
      return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />;
    case 'failed':
      return <XCircle className="w-5 h-5 text-red-500" />;
    case 'skipped':
      return <SkipForward className="w-5 h-5 text-gray-400" />;
    case 'pending':
    default:
      return <Circle className="w-5 h-5 text-gray-300" />;
  }
}

export function PipelineProgress({ steps, className = '' }: PipelineProgressProps) {
  if (!steps || steps.length === 0) {
    return null;
  }

  return (
    <div className={`bg-white border rounded-lg p-4 ${className}`}>
      <h3 className="text-sm font-semibold text-gray-600 mb-3">Pipeline Progress</h3>
      <div className="space-y-3">
        {steps.map((step, index) => {
          const label = STEP_LABELS[step.name] || step.name;
          const isLast = index === steps.length - 1;
          
          return (
            <div key={step.name} className="relative">
              {/* Connector line */}
              {!isLast && (
                <div 
                  className={`absolute left-[10px] top-6 w-0.5 h-6 ${
                    step.status === 'completed' ? 'bg-green-200' : 'bg-gray-200'
                  }`}
                />
              )}
              
              <div className="flex items-start gap-3">
                {/* Status icon */}
                <div className="flex-shrink-0 mt-0.5">
                  <StatusIcon status={step.status} />
                </div>
                
                {/* Step content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <span className={`text-sm font-medium ${
                      step.status === 'running' ? 'text-blue-700' :
                      step.status === 'failed' ? 'text-red-700' :
                      step.status === 'completed' ? 'text-gray-700' :
                      'text-gray-400'
                    }`}>
                      {label}
                    </span>
                    
                    {/* Duration badge for completed steps */}
                    {step.status === 'completed' && step.durationMs !== undefined && (
                      <span className="text-xs text-gray-400 font-mono">
                        {formatDuration(step.durationMs)}
                      </span>
                    )}
                  </div>
                  
                  {/* Progress bar for running step */}
                  {step.status === 'running' && (
                    <div className="mt-2">
                      <div className="w-full bg-gray-100 rounded-full h-1.5 overflow-hidden">
                        <div 
                          className="bg-blue-500 h-full rounded-full transition-all duration-300"
                          style={{ width: `${Math.max(5, step.progressPct)}%` }}
                        />
                      </div>
                      <div className="flex items-center justify-between mt-1">
                        {step.message && (
                          <span className="text-xs text-gray-500">{step.message}</span>
                        )}
                        {step.estimatedRemainingS !== undefined && step.estimatedRemainingS > 0 && (
                          <span className="text-xs text-gray-400">
                            {formatEstimate(step.estimatedRemainingS)}
                          </span>
                        )}
                        {!step.message && step.estimatedRemainingS === undefined && (
                          <span className="text-xs text-gray-400">{step.progressPct}%</span>
                        )}
                      </div>
                    </div>
                  )}
                  
                  {/* Message for failed step */}
                  {step.status === 'failed' && step.message && (
                    <p className="text-xs text-red-600 mt-1">{step.message}</p>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default PipelineProgress;
