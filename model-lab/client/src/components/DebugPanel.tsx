import React, { useEffect, useState } from 'react';
import { deriveStatusFingerprint, computeFingerprintHash } from '../lib/statusFingerprint';
import type { HelperStatusPayload } from '../lib/statusFingerprint';

interface DebugPanelProps {
    run: any; // Using any for flexibility with raw API payloads
}

export const DebugPanel: React.FC<DebugPanelProps> = ({ run }) => {
    const isDebug = import.meta.env.VITE_DEBUG_UI === '1';
    const [rawFingerprint, setRawFingerprint] = useState<string>('');
    const [hashFingerprint, setHashFingerprint] = useState<string>('');
    const [showRaw, setShowRaw] = useState(false);

    useEffect(() => {
        if (!run) return;

        // Map run object to helper payload
        const payload: HelperStatusPayload = {
            run_id: run.run_id,
            status: run.status,
            current_step: run.current_step,
            steps_completed: run.steps_completed,
            failed_step: run.failed_step,
            error_message: run.error_message,
            snapshot_source: run.snapshot_source,
            manifest_mtime: run.manifest_mtime,
            manifest_schema_version: run.manifest_schema_version
        };

        const raw = deriveStatusFingerprint(payload);
        setRawFingerprint(raw);

        computeFingerprintHash(raw).then(h => setHashFingerprint(h.substring(0, 8))); // Short hash
    }, [run]);

    if (!isDebug) return null;

    return (
        <div className="mt-8 p-4 bg-gray-900 border border-gray-700 rounded-lg text-xs font-mono text-gray-300">
            <h3 className="text-sm font-bold text-yellow-500 mb-2 uppercase">🛠 Mode B Debug Panel</h3>

            <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                    <span className="text-gray-500 block">Fingerprint (SHA-8)</span>
                    <span className="text-white font-bold bg-gray-800 px-1 rounded">{hashFingerprint}</span>
                </div>
                <div>
                    <span className="text-gray-500 block">Snapshot Source</span>
                    <span className={`px-1 rounded ${run.snapshot_source === 'manifest' ? 'text-green-400' : 'text-red-400'}`}>
                        {run.snapshot_source || 'unknown'}
                    </span>
                </div>
                <div>
                    <span className="text-gray-500 block">Manifest MTime</span>
                    <span>{run.manifest_mtime || 'N/A'}</span>
                </div>
                <div>
                    <span className="text-gray-500 block">Schema Version</span>
                    <span>v{run.manifest_schema_version || 1}</span>
                </div>
            </div>

            <div className="mb-2">
                <span className="text-gray-500 block">Raw Fingerprint String</span>
                <div className="bg-black p-2 rounded overflow-x-auto whitespace-pre text-gray-400 select-all">
                    {rawFingerprint}
                </div>
            </div>

            <button
                onClick={() => setShowRaw(!showRaw)}
                className="text-blue-400 hover:text-blue-300 underline mb-2"
            >
                {showRaw ? 'Hide Raw JSON' : 'Show Raw JSON'}
            </button>

            {showRaw && (
                <div className="bg-black p-4 rounded overflow-auto max-h-96">
                    <pre>{JSON.stringify(run, null, 2)}</pre>
                </div>
            )}
        </div>
    );
};
